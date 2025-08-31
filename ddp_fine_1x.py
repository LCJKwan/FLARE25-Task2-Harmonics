import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from torch.optim import AdamW, lr_scheduler
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss, FocalLoss

from utils.dataset import get_transforms, get_data_files
from model.AttnUNet5 import AttnUNet5
from utils.ddp_trainer import DDPTrainer

torch.multiprocessing.set_sharing_strategy('file_system')

class SoftDiceFocalLoss(torch.nn.Module):
    def __init__(self, include_background=True, softmax=True, weight=None, 
                 lambda_focal=1.0, lambda_dice=1.0, gamma=2.0):
        super().__init__()
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=softmax,
            weight=weight,
            soft_label=True)    # Use soft labels
        self.focal_loss = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            use_softmax=softmax,
            weight=weight)
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_dice = self.dice_loss(inputs, targets)
        l_focal = self.focal_loss(inputs, targets)
        return self.lambda_dice * l_dice + self.lambda_focal * l_focal

def main_worker(rank: int,
                world_size: int,
                model,
                train_params: dict,
                output_dir: str,
                comments: list):
    """
    Entry point for each spawned process.
    """
    try:
        # 1) Set the GPU device for this rank
        torch.cuda.set_device(rank)

        # 2) Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:29500',
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f'cuda:{rank}')
        )

        # 3) Only rank 0 creates output folder
        if rank == 0:
            full_output = output_dir
            os.makedirs(full_output, exist_ok=True)
        else:
            full_output = None

        # Datasets
        train_tf, val_tf = get_transforms(
            train_params['shape'],
            train_params['data_augmentation']['spatial'],
            train_params['data_augmentation']['intensity'],
            train_params['data_augmentation']['coarse'],
            soft=True)  # Use soft labels
        train_ds = Dataset(
            data=get_data_files(
                images_dir="data/nifti/train_gt/images",
                labels_dir="data/nifti/train_gt/softquant",
                extension='.nii.gz') * 8
            + get_data_files(
                images_dir="data/nifti/train_pseudo/images",
                labels_dir="data/nifti/train_pseudo/softquant",
                extension='.nii.gz'),
            transform=train_tf)
        val_ds = Dataset(
            data=get_data_files(
                images_dir="data/nifti/val/images",
                labels_dir="data/nifti/val/labels",
                extension='.nii.gz'),
            transform=val_tf)
        train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=train_params['batch_size'],
            sampler=train_sampler,
            num_workers=32,
            prefetch_factor=4,
            pin_memory=False,
            persistent_workers=True)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=False,
            persistent_workers=True)

        # Model, optimizer, scheduler, loss
        optimizer = AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'], eta_min=train_params['min_lr'])
        criterion = SoftDiceFocalLoss(  # Use soft labels
            include_background=True, 
            softmax=True, 
            weight=torch.tensor([0.05] + train_params["weights"], device=rank),
            lambda_focal=2,
            lambda_dice=1)


        # Initialize trainer and start
        trainer = DDPTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            train_params=train_params,
            output_dir=full_output,
            local_rank=rank,
            world_size=world_size,
            comments=comments)
        trainer.train(train_loader, val_loader, soft=True)

    except Exception as e:
        print(f"Rank {rank} crashed:", traceback.format_exc())
    finally:
        dist.destroy_process_group()


def get_comments(output_dir, train_params):
    return [
        f"{output_dir} - GT*8 + Soft Pseudo; 0.05 background loss weight",
        f"{train_params['shape']} shape, (2, 2, 1) patch embedding, k3 conv smooth after convtranspose (k3 merge), LayerNormTranspose", 
        f"SoftDiceFocal, 1-sample rand crop + augmentations",
        f"Spatial {train_params['data_augmentation']['spatial']}; Intensity {train_params['data_augmentation']['intensity']}; Coarse {train_params['data_augmentation']['coarse']}",
        "AU5 with depth-dropout, more spatial aug, class loss weight [1,2] by log space"
    ]


if __name__ == "__main__":
    # If needed:    pkill -f -- '--multiprocessing-fork'
    gpu_count = torch.cuda.device_count()
    architectures = ["AttnUNet5"]

    for architecture in architectures:
        model_params = json.load(open(f"configs/labellers/{architecture}/model.json"))
        train_params = json.load(open(f"configs/labellers/{architecture}/train1.json"))
        output_dir = f"output/Labeller/{architecture}-Pass1"
        comments = get_comments(f"{architecture}", train_params)

        print(f"Starting training for {architecture}...")
        if architecture == "AttnUNet5":
            model = AttnUNet5(model_params)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        try:
            mp.spawn(
                main_worker,
                args=(gpu_count, model, train_params, output_dir, comments),
                nprocs=gpu_count,
                join=True)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught in main process. Terminating children...")
            mp.get_context('spawn')._shutdown()
    
    
