import os
from monai.data import MetaTensor
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
    

class SumToLabeld(mt.MapTransform):
    def __init__(self, keys, output_key, weight1=127, weight2=128):
        super().__init__(keys)
        assert len(keys) == 2, "Must provide exactly 2 keys"
        self.output_key = output_key
        self.weight1 = weight1
        self.weight2 = weight2
        self.one_hot = mt.Compose([mt.AsDiscrete(to_onehot=14), mt.EnsureType(dtype=torch.uint8)])
        self.delete_keys = mt.DeleteItemsd(keys=keys)

    @torch.no_grad()
    def __call__(self, data):
        d = dict(data)
        a1 = d[self.keys[0]]
        a2 = d[self.keys[1]]

        a1_valid = a1.any(); a2_valid = a2.any()
        a1 = self.one_hot(a1)
        a2 = self.one_hot(a2)
        label = torch.zeros_like(a1)
        if a1_valid and a2_valid:
            a1.mul_(self.weight1)
            a2.mul_(self.weight2)
            label.add_(a1).add_(a2)
        elif a1_valid and not a2_valid:
            a1.mul_(self.weight1 + self.weight2)
            label.add_(a1)
        elif a2_valid and not a1_valid:
            a2.mul_(self.weight1 + self.weight2)
            label.add_(a2)
        else:
            raise ValueError("Both inputs are empty, cannot create label.")

        d[self.output_key] = MetaTensor(label, meta=a1.meta)
        d = self.delete_keys(d)
        return d


def get_pseudo_data(aladdin, blackbean, extension=".nii.gz"):
    aladdin = Path(aladdin)
    blackbean = Path(blackbean)
    data = []
    for img_path in aladdin.glob(f"*{extension}"):
        aladdin_path = aladdin / img_path.name
        blackbean_path = blackbean / img_path.name
        assert blackbean_path.exists(), f"Match for {img_path.name} not found in {blackbean}"
        data.append({
            "aladdin": str(aladdin_path),
            "blackbean": str(blackbean_path)
        })
    print(f"[INFO] found {len(data)} image/label data")
    return data



def process_dataset(aladdin, blackbean, out_dir, pixdim):
    # create output dirs
    os.makedirs(out_dir, exist_ok=True)

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["aladdin", "blackbean"], ensure_channel_first=True),
            mt.EnsureTyped(
                keys=["aladdin", "blackbean"],
                dtype=torch.uint8,
                track_meta=True),
            mt.ThresholdIntensityd(
                keys=["aladdin", "blackbean"],
                above=False,
                threshold=14,   # 14 classes
                cval=0),
            # mt.KeepLargestConnectedComponentd(
            #     keys=["aladdin", "blackbean"],
            #     independent=True,
            #     num_components=1),
            # mt.FillHolesd(keys=["aladdin", "blackbean"]),
            mt.Orientationd(keys=["aladdin", "blackbean"], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=["aladdin", "blackbean"],
                pixdim=pixdim,
                mode="nearest",
                lazy=True),
            SumToLabeld(
                keys=["aladdin", "blackbean"],
                output_key="label"),
            # mt.MeanEnsembled(
            #     keys=["aladdin", "blackbean"],
            #     output_key="label"),
            # QuantizeNormalized(keys=["label"]),
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
    dataset = Dataset(data=get_pseudo_data(aladdin, blackbean), transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=64,
        persistent_workers=True,
        prefetch_factor=32,
    )

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc=f"Creating Soft Labels"):
        pass

    return 



def process_gt(in_dir, out_dir, pixdim):
    # create output dirs
    os.makedirs(out_dir, exist_ok=True)

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], ensure_channel_first=True),
            mt.AsDiscreted(
                keys=["label"],
                to_onehot=14),  # 14 classes
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
        num_workers=50)

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc=f"Processing GT to Soft"):
        pass

    return 


if __name__ == "__main__":
    pixdim = (0.8, 0.8, 2.5)
    process_gt(
        "data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr",
        "data/nifti/train_gt/softquant",
        pixdim)
    process_dataset(
        "data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo",
        "data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_blackbean_flare22",
        "data/nifti/train_pseudo/softquant",
        pixdim)
    # shutil.copytree("data/nifti/train_pseudo/softquant", "data/nifti/train_pseudo/softiterative", dirs_exist_ok=True)