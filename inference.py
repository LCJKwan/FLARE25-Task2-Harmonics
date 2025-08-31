import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import json
import torch
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from monai.transforms import LoadImaged
from monai.metrics import compute_dice, compute_surface_dice
from monai.networks.utils import one_hot
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

# --- your model imports ---
from utils.RemoveSmall import RemoveSmallObjectsPerClassd
from model.AttnUNet6 import AttnUNet6

def get_image_label_pairs(images_dir, labels_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    pairs = []
    for img_path in images_dir.glob(f"*{extension}"):
        lbl_path = labels_dir / img_path.name
        if lbl_path.is_file():
            pairs.append({"img": str(img_path), "label": str(lbl_path)})
        else:
            print(f"[WARN] no label found for {img_path.name}")
    print(f"[INFO] found {len(pairs)} image/label pairs")
    return pairs

def get_pre_transforms(pixdim, intensities):
    upper, lower, mean, std = intensities
    spatial = mt.Compose([
        mt.LoadImaged(["img"], image_only=False, ensure_channel_first=True),
        mt.EnsureTyped(["img"], dtype=[torch.float32], track_meta=True),
        mt.Orientationd(["img"], axcodes="RAS", lazy=False),
        mt.Spacingd(["img"], pixdim=pixdim, mode="trilinear", lazy=False),
        mt.CenterSpatialCropd(["img"], roi_size=(256, 256, -1), lazy=False),
    ])
    intensity = mt.Compose([
        mt.ThresholdIntensityd(["img"], above=False, threshold=upper, cval=upper),
        mt.ThresholdIntensityd(["img"], above=True, threshold=lower, cval=lower),
        mt.NormalizeIntensityd(["img"], subtrahend=mean, divisor=std),
    ])
    return spatial, intensity

def get_post_transforms(pre_transforms, out_dir):
    return mt.Compose([
        mt.AsDiscreted(keys="pred", argmax=True),
        RemoveSmallObjectsPerClassd(keys=["pred"],
            min_sizes=[1e4, 1e3, 1e3, 1e3, 1e3, 1e3, 50, 100, 300, 100, 1000, 500, 500]),
        mt.Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="img",
            meta_keys="pred_meta_dict",
            orig_meta_keys="img_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True),
        mt.SaveImaged(keys="pred",
            output_dir=out_dir, 
            output_postfix="", 
            output_ext=".nii.gz", 
            resample=False,     # Invert already resamples
            separate_folder=False,
            output_dtype=torch.uint8,
            print_log=False)
    ])

# --- CPU-side post-processing function (must be top-level for pickling) ---
def cpu_post(pair, data, inference_config, num_classes):
    # Reconstruct inverter
    spatial_tf, _ = get_pre_transforms(
        inference_config["pixdim"], inference_config["intensities"]
    )
    inverter = get_post_transforms(spatial_tf, inference_config["out_dir"])

    # Prepare data for inversion
    pred_inv = inverter(data)["pred"].unsqueeze(0)

    # Load and one-hot label
    lbl_data = LoadImaged(["label"], ensure_channel_first=True)({"label": pair["label"]})
    label = lbl_data["label"].unsqueeze(0)

    pred_oh = one_hot(pred_inv, num_classes=num_classes)
    lbl_oh  = one_hot(label, num_classes=num_classes)

    dice_vals = compute_dice(
        y_pred=pred_oh, y=lbl_oh,
        include_background=False, ignore_empty=False
    ).numpy()
    # surf_vals = compute_surface_dice(
    #     y_pred=pred_oh, y=lbl_oh,
    #     include_background=False,
    #     class_thresholds=[1]*(num_classes-1)
    # ).numpy()

    data = mt.DeleteItemsd(keys=["img", "pred"])(data)
    lbl_data = mt.DeleteItemsd(keys=["label"])(lbl_data)

    return dice_vals, None#, surf_vals

@torch.inference_mode()
def run_and_score(
    chunk, inference_config, model, device,
    shared_metrics, gpu_id, autocast, n_cpu_workers, num_classes
):
    # Pre-build loader
    spatial_tf, intensity_tf = get_pre_transforms(
        inference_config["pixdim"], inference_config["intensities"]
    )
    loader = mt.Compose([spatial_tf, intensity_tf])

    # Inference + dispatch to CPU pool
    in_flight = set()
    precision = torch.bfloat16 if autocast else torch.float32
    with ProcessPoolExecutor(max_workers=n_cpu_workers) as executor:
        for pair in tqdm(chunk, desc=f"GPU {gpu_id}", unit="img"):
            try:
                # CPU → GPU prep
                data = loader({"img": pair["img"]})
                img = data["img"].to(device).unsqueeze(0)

                # GPU inference
                with torch.autocast("cuda", precision):
                    data["pred"] = sliding_window_inference(
                    img,
                    roi_size=inference_config["shape"],
                    sw_batch_size=inference_config.get("sw_batch_size", 1),
                    predictor=model,
                    overlap=inference_config.get("sw_overlap", 0.25),
                    mode="gaussian",
                    sw_device=device,
                    device=torch.device("cpu"),
                    buffer_steps=2
                ).cpu().squeeze(0)

            except Exception as e:
                print(f"[ERROR] GPU {gpu_id} failed on {pair['img']}: {e}")

            fut = executor.submit(cpu_post, pair, data, inference_config, num_classes)
            in_flight.add(fut)

        for f in as_completed(in_flight):
            dice_vals, surf_vals = f.result()
            shared_metrics.append((dice_vals, surf_vals))


def worker(
    gpu_id, chunks, inference_config,
    model_class, model_config, model_path,
    shared_metrics, autocast, n_cpu_workers, num_classes
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Build & load model
    model = model_class(model_config)
    state = torch.load(model_path, map_location=device, weights_only=True)

    # EMA model state dict handling
    state_keys = list(state.keys())
    for k in state_keys:
        if "module." in k:
            new_k = k
            while new_k[len("module."):] == "module.":
                new_k = new_k[len("module."):]
            state[new_k] = state.pop(k)
    if "n_averaged" in state_keys:
        del state["n_averaged"]

    model.load_state_dict(state)
    model.to(device).eval()

    # Run inference + CPU post
    run_and_score(
        chunk=chunks[gpu_id],
        inference_config=inference_config,
        model=model,
        device=device,
        shared_metrics=shared_metrics,
        gpu_id=gpu_id,
        autocast=autocast,
        n_cpu_workers=n_cpu_workers,
        num_classes=num_classes,
    )

if __name__ == "__main__":
    # --- configuration ---
    model_class     = AttnUNet6
    model_config    = json.load(open("configs/small/model.json", "r"))
    model_path      = "output/Small/AttnUNet6-600/model.pth"
    autocast        = True
    num_classes     = 14

    inference_config = {
        "pixdim": [1.6, 1.6, 2.5],
        "intensities": [295.0, -974.0, 95.958, 139.964],
        "shape": [256, 256, 64],
        "sw_batch_size": 8,
        "sw_overlap": 0.2,
        "out_dir": "archived_code/plots/labels/au6_official",
    }

    images_dir      = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images"
    labels_dir      = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels"

    # Prepare data & split
    all_pairs = get_image_label_pairs(images_dir, labels_dir)
    ngpus     = torch.cuda.device_count()
    np.random.shuffle(all_pairs)
    chunks    = np.array_split(all_pairs, ngpus)

    # Shared list for metrics
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    metrics = manager.list()

    # Decide how many CPU workers per GPU (e.g. total_cpus // ngpus)
    total_cpus = 64
    cpus_per_gpu = max(1, total_cpus // ngpus)

    # Spawn one process per GPU
    try:
        mp.spawn(
            fn=worker,
            args=(
                chunks,
                inference_config,
                model_class,
                model_config,
                model_path,
                metrics,
                autocast,
                cpus_per_gpu,
                num_classes
            ),
            nprocs=ngpus,
            join=True,
            daemon=False,  # ensure workers are not daemonic
            # No timeout argument is set, so workers can run indefinitely
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main process. Terminating children...")
        mp.get_context('spawn')._shutdown()

    # Aggregate & print
    dice_list, surf_list = zip(*list(metrics))
    dice_array = np.stack(dice_list, axis=0)
    # surf_array = np.stack(surf_list, axis=0)

    class_dices = np.nanmean(dice_array, axis=0).squeeze()
    # class_surfs = np.nanmean(surf_array, axis=0).squeeze()
    mean_dice   = class_dices.mean()
    # mean_surf   = class_surfs.mean()

    print("Per-class Dices:", class_dices)
    # print("Per-class Surface Dices:", class_surfs)
    print("Mean Dice:", mean_dice)
    # print("Mean Surface Dice:", mean_surf)

    # weights = list((1.01 - class_dices) / (1.01 - class_dices).mean())
    # print("Class weights:", [round(w, 3) for w in weights])
