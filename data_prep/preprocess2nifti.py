import os
from pathlib import Path
from tqdm import tqdm
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader


def get_dir_files(dir, split, extension=".nii.gz"):
    dir = Path(dir)
    if not dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir!r}")
    names = sorted(
        entry.name
        for entry in os.scandir(dir)
        if entry.is_file() and entry.name.endswith(extension)
    )
    if not names:
        raise RuntimeError(f"No '{extension}' files found in {dir!r}")
    return [
        {split: str(dir / name)}
        for name in names
    ]



def process_dataset(in_dir, out_dir, split, pixdim):
    # create output dirs
    os.makedirs(out_dir, exist_ok=True)

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=[split], ensure_channel_first=True),
            mt.Orientationd(keys=[split], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=[split],
                pixdim=pixdim,
                mode= "trilinear" if split=="image" else "nearest",
                lazy=True,
            ),
            mt.EnsureTyped(
                keys=[split],
                dtype=torch.float32 if split=="image" else torch.uint8,
                track_meta=True,
            ),
            mt.ThresholdIntensityd(
                keys=["label"],
                above=False,
                threshold=14,   # 14 classes
                cval=0,
            ) if split == "label" else mt.Identityd(keys=["image"]),
            mt.ThresholdIntensityd( # upper bound 99.5% from Aladdin5 + GT stats
                keys=["image"],
                above=False,
                threshold=295.0,
                cval=295.0,
            ) if split == "image" else mt.Identityd(keys=["label"]),
            mt.ThresholdIntensityd( # lower bound 0.5% from Aladdin5 + GT stats
                keys=["image"],
                above=True,
                threshold=-974.0, 
                cval=-974.0,
            ) if split == "image" else mt.Identityd(keys=["label"]),
            mt.NormalizeIntensityd( # z-score normalization from GT stats
                keys=["image"],
                subtrahend=95.958,
                divisor=139.964,
            ) if split == "image" else mt.Identityd(keys=["label"]),
            mt.SaveImaged(
                keys=["image"],
                output_dir=out_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.float32,
                print_log=False) if split == "image" else \
            mt.SaveImaged(
                keys=["label"],
                output_dir=out_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.uint8,
                print_log=False),
            mt.DeleteItemsd(keys=[split])
        ]
    )

    # build the MONAI dataset
    dataset = Dataset(data=get_dir_files(in_dir, split), transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=128,
        persistent_workers=True,
    )

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc=f"Processing {split}"):
        pass

    return 



if __name__ == "__main__":
    pixdim = (0.8, 0.8, 2.5)
    data_list = [
        (
            "data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr",
            "data/nifti/train_gt/images",
            "image", pixdim
        ),
        # (
        #     "data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr",
        #     "data/nifti/train_gt/labels",
        #     "label", pixdim
        # ),
        (
            "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images",
            "data/nifti/val/images",
            "image", pixdim
        ),
         (
            "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels",
            "data/nifti/val/labels",
            "label", pixdim
        ),
        (
            "data/FLARE-Task2-LaptopSeg/validation/Validation-Hidden-Images",
            "data/nifti/val/hidden",
            "image", pixdim
        ),
        (
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr",
            "data/nifti/train_pseudo/images",
            "image", pixdim
        ),
        (
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo",
            "data/nifti/train_pseudo/aladdin5",
            "label", pixdim
        ),
        (
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_blackbean_flare22",
            "data/nifti/train_pseudo/blackbean",
            "label", pixdim
        )
    ]

    for data in data_list:
        process_dataset(*data)
    
