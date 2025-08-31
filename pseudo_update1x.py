import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import argparse
import json
import torch
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from monai.data import MetaTensor

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
torch.multiprocessing.set_sharing_strategy('file_system')

# --- your model imports ---
from utils.RemoveSmall import RemoveSmallObjectsPerClassd
from model.AttnUNet5 import AttnUNet5


def get_pseudo_data(images, aladdin, blackbean, extension=".nii.gz"):
    images = Path(images)
    aladdin = Path(aladdin)
    blackbean = Path(blackbean)
    data = []
    for img_name in images.glob(f"*{extension}"):
        img_name = img_name.name
        image_path = images / img_name
        aladdin_path = aladdin / img_name
        blackbean_path = blackbean / img_name
        assert blackbean_path.exists(), f"Match for {img_name} not found in {blackbean}"
        data.append({
            "img": str(image_path),
            "aladdin": str(aladdin_path),
            "blackbean": str(blackbean_path)
        })
    print(f"[INFO] found {len(data)} image/label data")
    return data


# --- CPU-side post-processing function ---
@torch.no_grad()
def cpu_post(data, inference_config):
    prep_tf = mt.Compose([
        mt.AsDiscreted(keys=["pred"], argmax=True),
        mt.EnsureTyped(keys=["pred"], dtype=torch.uint8),
        mt.LoadImaged(keys=["aladdin", "blackbean"], ensure_channel_first=True),
    ])
    post_tf = mt.Compose([
        mt.DeleteItemsd(keys=["aladdin", "blackbean", "fg"]),
        mt.SaveImaged(
            keys=["pred"],
            output_dir=inference_config["output_dir"],
            output_postfix="",
            output_ext=".nii.gz",
            separate_folder=False,
            output_dtype=torch.uint8,
            print_log=False),
        mt.DeleteItemsd(keys=["pred"])
    ])
    remove_small = mt.Compose([
        RemoveSmallObjectsPerClassd(
            keys=["pred"],
            min_sizes=[1e4, 1e3, 1e3, 1e3, 1e3, 1e3, 100, 100, 500, 100, 1e3, 1e3, 1e3]),
        mt.AsDiscreted(keys=["pred"], to_onehot=14)
    ])
    to_uint8 = mt.EnsureTyped(keys=["pred", "aladdin", "blackbean"], dtype=torch.uint8)
    crop = mt.CropForegroundd(keys=["pred", "aladdin", "blackbean"], 
                              source_key="fg",
                              margin=(25, 25, 8), # 2cm margin
                              allow_smaller=True)

    # Prepare
    data = prep_tf(data)
    pred = torch.cat([  # Default is background class
        torch.ones((1, *data["pred"].shape[1:]), dtype=torch.uint8).mul_(255),
        torch.zeros((13, *data["pred"].shape[1:]), dtype=torch.uint8)],
        dim=0)  # 14 classes, 1 background + 13 foreground

    # Get foreground
    data["fg"] = (data["aladdin"] > 0) | (data["blackbean"] > 0)
    data = crop(data)

    offset = data.get("foreground_start_coord", (0, 0, 0))  # this gives ROI origin indices
    slices = tuple(slice(start, start + size) for start, size in zip(offset, data["pred"].shape[-len(offset):]))

    # Zero out foreground
    pred[(...,) + slices] = 0

    # Remove small objects
    data = remove_small(data)

    # Aladdin Blackbean adding
    aladdin_valid = data["aladdin"].any()
    blackbean_valid = data["blackbean"].any()
    if aladdin_valid and blackbean_valid:
        data = mt.AsDiscreted(keys=["aladdin", "blackbean"], to_onehot=14)(data)
        data = to_uint8(data)
        data["aladdin"].mul_(85)
        data["blackbean"].mul_(85)
        data["pred"].mul_(85)
        pred[(...,) + slices].add_(data["aladdin"])
        pred[(...,) + slices].add_(data["blackbean"])
    elif aladdin_valid and (not blackbean_valid):
        data = mt.AsDiscreted(keys=["aladdin"], to_onehot=14)(data)
        data = to_uint8(data)
        data["aladdin"].mul_(127)
        data["pred"].mul_(128)
        pred[(...,) + slices].add_(data["aladdin"])
    elif (not aladdin_valid) and blackbean_valid:
        data = mt.AsDiscreted(keys=["blackbean"], to_onehot=14)(data)
        data = to_uint8(data)
        data["blackbean"].mul_(127)
        data["pred"].mul_(128)
        pred[(...,) + slices].add_(data["blackbean"])
    else:
        data = to_uint8(data)
        data["pred"].mul_(255)

    # Add prediction
    pred[(...,) + slices].add_(data["pred"])


    # Save
    data["pred"] = MetaTensor(pred, meta=data["blackbean"].meta)
    data = post_tf(data)
    return 

@torch.no_grad()
def run_and_save(
    chunk, inference_config, model, device,
    gpu_id, n_cpu_workers, max_prefetch
):
    # Pre-build loader
    dataloader = DataLoader(
        Dataset(data=chunk, transform=mt.LoadImaged(["img"], ensure_channel_first=True)),
        batch_size=1,
        num_workers=10,
        pin_memory=False,
    )
    deleter = mt.DeleteItemsd(["img"])

    # Inference + dispatch to CPU pool
    in_flight = set()
    autocast = torch.bfloat16 if inference_config["autocast"] else torch.float32
    with ProcessPoolExecutor(max_workers=n_cpu_workers) as executor:
        for data in tqdm(dataloader, desc=f"GPU {gpu_id}"):
            try:
                # CPU → GPU prep
                img = data["img"].to(device, non_blocking=True)

                # GPU inference
                with torch.autocast("cuda", autocast):
                    data["pred"] = sliding_window_inference(
                        img,
                        roi_size=inference_config["shape"],
                        sw_batch_size=inference_config.get("sw_batch_size", 1),
                        predictor=model,
                        overlap=inference_config["sw_overlap"],
                        mode="gaussian",
                        sw_device=device,
                        device=torch.device("cpu"),
                        buffer_steps=4,
                    ).cpu().squeeze(0).numpy()

            except Exception as e:
                print(f"[ERROR] GPU {gpu_id} failed: {e}")

            # Done with image
            data = deleter(data)

            # 3) submit to CPU pool
            fut = executor.submit(cpu_post, data, inference_config)
            in_flight.add(fut)

            # 4) if we've queued >= max_prefetch, wait for at least one to finish
            if len(in_flight) >= (n_cpu_workers + max_prefetch):
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()

        # drain remaining futures
        for f in as_completed(in_flight):
            f.result()


def worker(
    gpu_id, chunks, inference_config,
    model_class, model_config, model_path,
    n_cpu_workers, max_prefetch
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Build & load model
    model = model_class(model_config)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # Run inference + CPU post
    run_and_save(
        chunk=chunks[gpu_id],
        inference_config=inference_config,
        model=model,
        device=device,
        gpu_id=gpu_id,
        n_cpu_workers=n_cpu_workers,
        max_prefetch=max_prefetch
        )

if __name__ == "__main__":
    # --- configuration ---
    parser = argparse.ArgumentParser(description="Update soft pseudo labels inference.")
    parser.add_argument("--config", type=str, help="Path to the inference configuration file.")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model weights.")
    args = parser.parse_args()
    inference_config = json.load(open(args.config, "r"))
    os.makedirs(inference_config["output_dir"], exist_ok=True)

    if inference_config["model_class"] == "AttnUNet5":
        model_class = AttnUNet5
    model_config    = json.load(open(inference_config["model_config"], "r"))
    model_path      = args.model_path

    # Prepare data & split
    all_pairs = get_pseudo_data(
        images="data/nifti/train_pseudo/images",
        aladdin="data/nifti/train_pseudo/aladdin5",
        blackbean="data/nifti/train_pseudo/blackbean")
    ngpus     = torch.cuda.device_count()
    np.random.shuffle(all_pairs)
    chunks    = np.array_split(all_pairs, ngpus)

    # Decide how many CPU workers per GPU (e.g. total_cpus // ngpus)
    cpus_per_gpu = 30
    max_prefetch = 8

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
                cpus_per_gpu,
                max_prefetch
            ),
            nprocs=ngpus,
            join=True,
            daemon=False,  # ensure workers are not daemonic
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main process. Terminating children...")
        mp.get_context('spawn')._shutdown()

    print("Soft pseudo labels updated successfully.")