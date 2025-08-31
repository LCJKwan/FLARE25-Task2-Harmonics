import os
from pathlib import Path
from tqdm import tqdm
import torch
import monai.transforms as mt
from monai.data import Dataset, DataLoader
from monai.config import KeysCollection

class QuantizeNormalized(mt.MapTransform):
    """
    Dictionary-based MONAI transform to quantize a float tensor along dim=0 so that each slice sums to 255 (uint8),
    preserving original proportions as closely as possible.

    Args:
        keys: Key or list of keys in the input dictionary whose values are torch.Tensors to be quantized.
    """

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys)

    def __call__(self, data):
        # make a shallow copy so we don't modify the original dict
        d = dict(data)
        for key in self.keys:
            x = d[key]
            if not torch.is_tensor(x):
                raise TypeError(f"QuantizeNormalized: expected torch.Tensor for key '{key}', got {type(x)}")
            d[key] = self._quantize(x)
        return d

    @staticmethod
    def _quantize(x: torch.Tensor) -> torch.Tensor:
        # 1) compute channel‐sums
        sums = x.sum(dim=0, keepdim=True)
        sums[sums == 0] = 1.0  # avoid division by zero

        # 2) scale each channel so sum→255
        scaled = x.mul(255.0).div_(sums)

        # 3) get integer floor via a single cast and compute residuals
        floors = scaled.to(torch.uint8)          # float→uint8 is a truncation cast :contentReference[oaicite:1]{index=1}
        residuals = scaled - floors.float()

        # 4) compute how many “ones” to distribute per spatial location
        deficits = (255 - floors.sum(dim=0, keepdim=True))

        # 5) vectorize the “largest‐residual” selection
        C = x.size(0)
        # flatten spatial dims into one axis
        res_flat = residuals.view(C, -1)         # shape [C, N]
        def_flat = deficits.view(-1)            # shape [N]

        # single sort along channel axis
        _, idx_flat = res_flat.sort(dim=0, descending=True)  # one sort call :contentReference[oaicite:2]{index=2}

        # build a mask in sorted order: for each pixel j, top def_flat[j] channels get +1
        dr = torch.arange(C, device=x.device).view(C, 1)
        mask_sorted = dr < def_flat.unsqueeze(0)            # shape [C, N]

        # scatter the mask back to original channel positions
        mask_flat = torch.zeros_like(mask_sorted)
        mask_flat.scatter_(0, idx_flat, mask_sorted)

        # reshape mask to [C, ...] and form final result
        mask = mask_flat.view_as(x).to(torch.uint8)
        return (floors + mask).to(torch.uint8)
    

def process_pseudo(in_dir, out_dir, pixdim):
    # create output dirs
    os.makedirs(out_dir, exist_ok=True)

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], ensure_channel_first=True, image_only=False),
            mt.EnsureTyped(
                keys=["label"],
                dtype=torch.float32,
                track_meta=True),
            mt.Orientationd(keys=["label"], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=["label"],
                pixdim=pixdim,
                mode="trilinear",
                lazy=True),
            QuantizeNormalized(keys=["label"]),
            mt.SaveImaged(
                keys=["label"],
                output_dir=out_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.uint8,
                print_log=False),
            mt.DeleteItemsd(keys=["label"])
        ]
    )

    # build the MONAI dataset
    dir = Path(in_dir)
    if not dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir!r}")
    names = sorted(
        entry.name
        for entry in os.scandir(dir)
        if entry.is_file() and entry.name.endswith(".nii.gz"))
    dataset = Dataset(data=[{"label": str(dir / name)} for name in names], transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=40,
        prefetch_factor=32,
        pin_memory=False)

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc=f"Processing Pseudo to Soft"):
        pass

    return 

if __name__ == "__main__":
    input_dir = "data/nifti/train_pseudo/pseudo1x"
    output_dir = "data/small/train_pseudo/pseudo1x"

    process_pseudo(input_dir, output_dir, pixdim=(1.6, 1.6, 2.5))