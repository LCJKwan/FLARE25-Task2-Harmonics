import argparse
import json
import torch
import numpy as np
import monai.transforms as mt
from monai.data import Dataset
from monai.inferers import sliding_window_inference
from pathlib import Path
import os
from tqdm import tqdm
from monai.transforms import Transform
from monai.data import MetaTensor
from skimage.morphology import remove_small_objects
from nibabel.orientations import aff2axcodes

def get_image_files(images_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    image_dicts = [
        {"img": str(entry.path)}
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    ]
    return image_dicts


class RemoveSmallObjectsPerClass(Transform):
    def __init__(self,
            labels=list(range(1, 14)),
            min_sizes=[1e4, 1e3, 1e3, 1e3, 1e3, 1e3, 50, 100, 300, 100, 1000, 500, 500],
            connectivity=1):
        self.labels = labels
        self.min_sizes = min_sizes
        self.conn = connectivity

    def __call__(self, img):
        for lbl, ms in zip(self.labels, self.min_sizes):
            mask = (img == lbl)
            if mask.any():
                cleaned_mask = remove_small_objects(mask, min_size=ms, connectivity=self.conn)
                img[mask & (~cleaned_mask)] = 0
        return img 


def monai_target_shape(spatial_shape, src_pixdim, dst_pixdim):
    return tuple(int(round(s * p_src / p_dst)) for s, p_src, p_dst in zip(spatial_shape, src_pixdim, dst_pixdim))

def _center_crop(x: torch.Tensor, roi_size):
    D, H, W = x.shape[1:]
    t = []
    s = []
    for dim_len, req in zip((D,H,W), roi_size):
        size = dim_len if (req in (-1, None)) else min(dim_len, int(req))
        start = (dim_len - size) // 2
        # if odd remainder, MONAI effectively gives the extra voxel to the end
        end = start + size
        t.append(size)
        s.append(slice(start, end))
    cropped = x[:, s[0], s[1], s[2]]
    return cropped, (s[0], s[1], s[2])


def _insert_with_slices(full_shape, patch: torch.Tensor, crop_slices):
    out = torch.zeros(full_shape, dtype=patch.dtype, device=patch.device)
    out[(slice(None),) + crop_slices] = patch
    return out

def _clamp_and_norm_(x: torch.Tensor, lo, hi, mean, std):
    x.clamp_(min=lo, max=hi).sub_(mean).div_(std)
    return x



@torch.inference_mode()
def run_inference(args, inference_config):
    # Load the model
    model = torch.jit.load("model_cpu.pth", map_location="cpu")
    model.eval().to(args.device)

    # Create dataset and dataloader
    upper, lower, mean, std = inference_config["intensities"]
    pixdim = inference_config["pixdim"]
    loader = mt.Compose([
        mt.LoadImaged(["img"], image_only=False, ensure_channel_first=True),
        mt.EnsureTyped(["img"], dtype=torch.float32, track_meta=True),
        mt.Orientationd(["img"], axcodes="RAS")])
    preprocess = mt.Compose([
        mt.Spacingd(["img"], pixdim=pixdim, mode="trilinear", dtype=None), 
        mt.CenterSpatialCropd(["img"], roi_size=inference_config["max_shape"]), 
        mt.ThresholdIntensityd(["img"], above=False, threshold=upper, cval=upper),
        mt.ThresholdIntensityd(["img"], above=True, threshold=lower, cval=lower),
        mt.NormalizeIntensityd(["img"], subtrahend=mean, divisor=std)])
    remove_small = RemoveSmallObjectsPerClass()
    invert_all = mt.Invertd(
            keys="pred",
            transform=mt.Compose([loader, preprocess]),
            orig_keys="img",
            meta_keys="pred_meta_dict",
            orig_meta_keys="img_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True)
    saver = mt.SaveImaged(keys="pred",
            output_dir=args.output_dir, 
            output_postfix="", 
            output_ext=".nii.gz", 
            resample=False,     # Invert already resamples
            separate_folder=False,
            output_dtype=np.uint8,
            print_log=False)
    
    dataset = Dataset(
        data=get_image_files(args.input_dir), 
        transform=loader)
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    for data in tqdm(dataset, desc="Inference"):
        orig_pixdim = data["img"].meta["pixdim"][1:4].tolist()
        ss = [s*p1/p2 for s, p1, p2 in zip(data["img"].shape[1:], orig_pixdim, pixdim)]
        manual_invert = ss[0]*ss[1]*ss[2]*data["img"].numel() > 1.5e15

        if manual_invert:
            img_mt: MetaTensor = data["img"]  # (C,D,H,W), in RAS
            img_meta = img_mt.meta

            orig_shape_ras = tuple(img_mt.shape[1:])
            src_pixdim = img_meta["pixdim"][1:4].tolist()

            # Resample to target spacing on CPU
            resampled_shape = monai_target_shape(orig_shape_ras, src_pixdim, inference_config["pixdim"])
            img_resampled = torch.nn.functional.interpolate(
                img_mt.unsqueeze(0), size=resampled_shape, mode="trilinear", align_corners=False
            ).squeeze(0)

            # Center crop + intensity normalize (in-place)
            img_cropped, crop_slices = _center_crop(img_resampled, inference_config["max_shape"])
            _clamp_and_norm_(img_cropped, lower, upper, mean, std)

            # Sliding-window inference (send only crop to device)
            pred_crop = torch.argmax(sliding_window_inference(
                img_cropped.to(args.device, non_blocking=True).unsqueeze(0),
                roi_size=inference_config["shape"],
                sw_batch_size=inference_config["sw_batch_size"],
                predictor=model,
                overlap=inference_config["sw_overlap"],
                mode="gaussian",
            ).cpu().squeeze(0), dim=0, keepdim=True).numpy().astype(np.uint8) 

            # Postproc (skimage on CPU)
            pred_crop = torch.from_numpy(remove_small(pred_crop))

            # Invert: crop -> full resampled canvas -> original RAS grid -> original orientation
            pred_full_resampled = _insert_with_slices((1, *resampled_shape), pred_crop, crop_slices)
            pred_orig_ras = torch.nn.functional.interpolate(
                pred_full_resampled.unsqueeze(0), size=orig_shape_ras, mode="nearest"
            ).squeeze(0).numpy().astype(np.uint8)

            pred_mt = MetaTensor(pred_orig_ras, meta=img_meta.copy())
            pred_mt = mt.Orientation(axcodes=aff2axcodes(img_meta["original_affine"]))(pred_mt)
            saver({"pred": pred_mt, "pred_meta_dict": pred_mt.meta.copy()})

        else:
            data = preprocess(data)
            data["pred"] = torch.argmax(sliding_window_inference(
                        data["img"].to(args.device, non_blocking=True).unsqueeze(0),
                        roi_size=inference_config['shape'],
                        sw_batch_size=inference_config.get('sw_batch_size', 1),
                        predictor=lambda x: model(x),
                        overlap=inference_config.get('sw_overlap', 0.25),
                        mode="gaussian").cpu().squeeze(0), dim=0, keepdim=True).numpy().astype(np.uint8)
            data["pred"] = remove_small(data["pred"])

            data["pred_meta_dict"] = data["img"].meta.copy()
            data["pred"] = MetaTensor(data["pred"], meta=data["pred_meta_dict"])
            data = invert_all(data)
            saver(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=r'./inputs', help='dir of output')
    parser.add_argument('--output_dir', type=str, default=r'./outputs', help='dir of output')
    parser.add_argument('--device', type=str, default='cpu', help='device to run inference on')
    args = parser.parse_args()

    inference_config = json.load(open('./inference_config.json', 'r'))
    run_inference(args, inference_config)