"""
Production Training Engine for ELMDE.
Supports mixed precision, AdamW optimization, cosine learning rate schedules,
gradient clipping, CORN ordinal losses, and checkpoint management.
"""

from typing import Any, Callable, Dict, Optional
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import CORNLoss
from ..nn.model import ELMDEModel
from ..utils.device import get_optimal_device_and_dtype


class ELMDETrainer:
    """Production Trainer for fine-tuning and calibrating ELMDE models."""
    def __init__(
        self,
        model: ELMDEModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        total_epochs: int = 3,
        grad_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        device: Optional[torch.device] = None
    ):
        self.device, self.dtype = get_optimal_device_and_dtype() if device is None else (device, torch.float32)
        self.model = model.to(self.device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.total_epochs = total_epochs
        self.grad_clip = grad_clip_norm
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 1. Optimizer: AdamW with weight decay
        no_decay = ["bias", "norm", "contractive_scale"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # 2. Cosine Annealing Learning Rate Scheduler
        total_steps = len(train_dataloader) * total_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7)

        # 3. Loss function
        self.corn_loss_fn = CORNLoss(num_classes=model.config.num_ordinal_classes)

        # 4. Scaler for mixed precision
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes one training epoch across the dataset."""
        self.model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch.get("target_ordinal")
            if targets is not None:
                targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # Forward ordinal
                class_probs, cum_surv, metrics = self.model.forward_ordinal(input_ids)
                aux_loss = metrics.get("aux_loss", torch.tensor(0.0, device=self.device))

                # Recover cond_probs for CORN loss
                cond_probs = torch.empty_like(cum_surv)
                cond_probs[:, 0] = cum_surv[:, 0]
                for k in range(1, cum_surv.shape[-1]):
                    cond_probs[:, k] = cum_surv[:, k] / (cum_surv[:, k - 1] + 1e-9)
                cond_probs = torch.clamp(cond_probs, 1e-7, 1.0 - 1e-7)

                task_loss = self.corn_loss_fn(cond_probs, targets) if targets is not None else torch.tensor(0.0)
                loss = task_loss + aux_loss

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            # Tracking
            batch_size = input_ids.shape[0]
            total_loss += loss.item() * batch_size
            total_aux_loss += aux_loss.item() * batch_size
            total_samples += batch_size

            if targets is not None:
                preds = torch.argmax(class_probs, dim=-1)
                correct_predictions += (preds == targets).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = float(correct_predictions) / max(total_samples, 1)
        return {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "train_aux_loss": total_aux_loss / max(total_samples, 1),
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluates model performance on the validation split."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_confidences = []
        all_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch.get("target_ordinal")
                if targets is not None:
                    targets = targets.to(self.device)

                class_probs, cum_surv, metrics = self.model.forward_ordinal(input_ids)
                preds = torch.argmax(class_probs, dim=-1)
                confs, _ = torch.max(class_probs, dim=-1)

                batch_size = input_ids.shape[0]
                total_samples += batch_size

                if targets is not None:
                    is_correct = (preds == targets)
                    correct_predictions += is_correct.sum().item()
                    loss_vec = (1.0 - is_correct.float()).cpu().numpy()
                    all_losses.extend(loss_vec.tolist())
                    all_confidences.extend(confs.cpu().numpy().tolist())

        val_acc = float(correct_predictions) / max(total_samples, 1)
        return {
            "val_accuracy": val_acc,
            "val_samples": total_samples,
            "confidences": all_confidences,
            "losses": all_losses
        }

    def train_and_save(self, save_filename: str = "elmde_production.pt") -> str:
        """Runs the full training procedure across all epochs and saves the final checkpoint."""
        print(f"[*] Starting ELMDE Training on {self.device} (AMP={self.use_amp})...")
        best_val_acc = 0.0
        best_path = os.path.join(self.checkpoint_dir, save_filename)

        for epoch in range(1, self.total_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            print(f"    Epoch {epoch}/{self.total_epochs} | Train Loss: {train_metrics['train_loss']:.4f} | Accuracy: {train_metrics['train_accuracy']*100:.2f}%")

            val_metrics = self.evaluate()
            if val_metrics:
                print(f"    Validation | Accuracy: {val_metrics['val_accuracy']*100:.2f}%")
                if val_metrics["val_accuracy"] >= best_val_acc:
                    best_val_acc = val_metrics["val_accuracy"]
                    self.model.save_pretrained(self.checkpoint_dir, filename=save_filename)
            else:
                self.model.save_pretrained(self.checkpoint_dir, filename=save_filename)

        print(f"[*] Training Complete! Saved checkpoint to: {best_path}")
        return best_path
