import os
import json
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import monai.metrics as mm
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

class DDPTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler,
        train_params: dict,
        output_dir: str,
        local_rank: int = 0,
        world_size: int = 1,
        comments: list = None,
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.comments = comments or []
        self.train_params = train_params
        self.output_dir = output_dir

        # Device for this process (use local_rank directly)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        # Wrap in DDP if using multiple GPUs
        model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[self.local_rank], 
                output_device=self.local_rank, broadcast_buffers=False)
        else:
            self.model = model

        # Optimizations
        if train_params.get('autocast', False):
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
        if train_params.get("compile", False):
            self.model = torch.compile(self.model)

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.precision = torch.bfloat16 if train_params.get("autocast", False) else torch.float32

        # Only rank 0 writes metrics
        self.num_classes = train_params['num_classes']
        self.dice_metric = mm.DiceMetric(include_background=True, 
                                         ignore_empty=False,
                                         reduction='mean_batch')
        if self.local_rank == 0:
            self.train_losses = []
            self.val_losses = []
            self.val_metrics = {'dice': [], 'class_dice': []}
            self.best_results = {}
            self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.start_time = None
            self.class_names = ["Liver", "Right kidney", "Spleen", "Pancreas", 
                                "Aorta", "Inferior Vena Cava", "Right Adrenal Gland", "Left Adrenal Gland",
                                "Gallbladder", "Esophagus", "Stomach", "Duodenum", "Left kidney"]

    def train(self, train_loader, val_loader=None, soft=False):
        if self.local_rank == 0:
            self.start_time = time.time()

        epochs = self.train_params['epochs']
        agg_steps = self.train_params['aggregation']

        for epoch in range(epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            running_loss = 0.0
            grad_norm = torch.tensor(0.0, device=self.device)

            loop = tqdm.tqdm(train_loader,
                             desc=f"[Rank {self.local_rank}] Epoch {epoch+1}/{epochs}",
                             disable=(self.local_rank!=0))
            self.optimizer.zero_grad()

            for i, batch in enumerate(loop):
                imgs = batch['image'].to(self.device, non_blocking=True)
                masks = batch['label'].to(self.device, non_blocking=True)
                if soft:
                    masks = masks.float() / 255.0  # Convert to float on GPU
                else:
                    masks = one_hot(masks, num_classes=self.num_classes)

                with torch.autocast(device_type='cuda', dtype=self.precision):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)
                loss.backward()
                running_loss += loss.item()

                if ((i + 1) % agg_steps == 0) or (i + 1 == len(train_loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.local_rank == 0:
                    loop.set_postfix({'Norm': grad_norm.item(), 'Loss': loss.item()})

            self.scheduler.step()

            val_loss, metrics = self.evaluate(val_loader, one_hot_loss=True)
            if self.world_size > 1:
                torch.cuda.synchronize(self.device)
                dist.barrier()
            if self.local_rank == 0 and val_loader is not None:
                metrics["dice"] = float(sum(metrics["class_dice"][1:]) / len(metrics["class_dice"][1:]))
                self.train_losses.append(running_loss / len(train_loader))
                self.val_losses.append(val_loss)
                self.val_metrics['dice'].append(metrics['dice'])
                self.val_metrics['class_dice'].append(metrics['class_dice'])
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {self.train_losses[-1]:.5f} | "
                      f"Val Loss: {val_loss:.5f} | "
                      f"Val Dice: {metrics['dice']:.5f}")
                self.plot_results()
                self.save_checkpoint(epoch, metrics)


    @torch.no_grad()
    def evaluate(self, data_loader, one_hot_loss=False):
        self.model.eval()
        # local accumulators
        loss_sum = torch.tensor(0.0, device=self.device)
        sample_count = torch.tensor(0, device=self.device)
        # reset MONAI dice
        self.dice_metric.reset()

        # loop over your shard
        for batch in tqdm.tqdm(
            data_loader,
            desc=f"[Rank {self.local_rank}] Validation",
            disable=(self.local_rank != 0),
        ):
            imgs = batch['image'].to(self.device, non_blocking=True)
            masks = batch['label'].to(self.device, non_blocking=True)
            if one_hot_loss:    # Convert first
                masks = one_hot(masks, num_classes=self.num_classes)
            B = imgs.size(0)
            sample_count += B

            # sliding window inference as before
            with torch.autocast(device_type='cuda', dtype=self.precision):
                aggregated = sliding_window_inference(
                    imgs,
                    roi_size=self.train_params['shape'],
                    sw_batch_size=self.train_params.get('sw_batch_size', 1),
                    predictor=lambda x: self.model(x),
                    overlap=self.train_params.get('sw_overlap', 0.25),
                    mode="gaussian",
                    buffer_steps=None
                )

                loss = self.criterion(aggregated, masks)
            # accumulate loss weighted by batch size
            loss_sum += loss.item() * B

            # one‐hot encode and update dice
            preds = one_hot(
                torch.argmax(aggregated, dim=1, keepdim=True),
                num_classes=self.num_classes
            )
            if not one_hot_loss:    # Convert later
                masks = one_hot(masks, num_classes=self.num_classes)
            self.dice_metric(y_pred=preds, y=masks)

        # Aggregate loss and sample count across all ranks
        if self.world_size > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)

        total_loss = loss_sum.item() / max(sample_count.item(), 1)
        total_dice = list(self.dice_metric.aggregate().cpu().numpy())
        total_dice = [float(d) for d in total_dice]  # Convert to float for JSON serialization
        return total_loss, {'class_dice': total_dice}

    def save_checkpoint(self, epoch: int, val_metrics: dict):
        # Save last
        state_dict = (self.model.module.state_dict()
                if isinstance(self.model, DDP) else self.model.state_dict())
        torch.save(state_dict, os.path.join(self.output_dir, 'model.pth'))
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # Save best
        if self.val_metrics['dice'][-1] == max(self.val_metrics['dice']):
            torch.save(self.model.module.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            self.best_results = {
                'epoch': epoch,
                'train_loss': self.train_losses[-1],
                'val_loss': self.val_losses[-1],
                'val_metrics': val_metrics
            }

        # Write summary
        elapsed = time.time() - self.start_time
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write(f"Model size: {self.model_size/1e6:.2f}M\n")
            f.write(f"Training time: {int(hrs):02}:{int(mins):02}:{int(secs):02}\n\n")
            f.write(f"Epoch {epoch+1} results:\n")
            f.write(f"Train Loss: {self.train_losses[-1]:.5f}; Val Loss: {self.val_losses[-1]:.5f}; Val Dice: {self.val_metrics['dice'][-1]:.5f}\n\n")
            for c in self.comments:
                f.write(c + "\n")
            f.write(f"\nModel params: {json.dumps(self.model.module.model_params, indent=4)}\n")
            f.write(f"\nTrain params: {json.dumps(self.train_params, indent=4)}\n")
            f.write(f"\nBest results: {json.dumps(self.best_results, indent=4)}\n")

    def plot_results(self):
        epochs = range(1, len(self.train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        ax1.plot(epochs, self.train_losses, label='Train')
        ax1.plot(epochs, self.val_losses, label='Val')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.set_title('Loss')

        # Dice curve
        ax2.plot(epochs, self.val_metrics['dice'], label='Val Dice')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
        ax2.legend(); ax2.set_title('Validation Dice')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close(fig)

        # Plot class dice
        plt.figure(figsize=(12, 6))
        class_dice = np.array(self.val_metrics["class_dice"]).transpose()[1:].tolist()
        for name, dice in zip(self.class_names, class_dice):
            plt.plot(dice, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.title("Dice Score for Each Organ over Training")
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_dice.png'))
        plt.close()
