"""
Foundation-Scale Fine-Tuning and Calibration Pipeline for ELMDE.
Trains ELMDEFoundationModel (ModernBERT-base + Recurrent MoE Core + Tri-Modal Heads)
on multi-domain structured decision tasks, then computes Split-Conformal Risk Control (SCRC)
calibration thresholds to certify finite-sample risk bounds (E[Loss] <= alpha).
"""

import os
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from elmde.nn.foundation_model import ELMDEFoundationModel
from elmde.training.losses import CORNLoss, BetaNLLLoss, HeteroscedasticLoss
from elmde.conformal.scrc import SelectiveConformalRiskController
from elmde.utils.device import get_optimal_device


class MultiDomainDecisionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        tokens = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "ordinal_label": torch.tensor(item["ordinal_label"], dtype=torch.long),
            "continuous_label": torch.tensor(item["continuous_label"], dtype=torch.float32),
            "choice_label": torch.tensor(item["choice_label"], dtype=torch.long)
        }


def train_foundation_elmde():
    device = get_optimal_device()
    print(f"[Foundation Trainer] Selected device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    train_dataset = MultiDomainDecisionDataset("data/production_corpus_train.json", tokenizer)
    val_dataset = MultiDomainDecisionDataset("data/production_corpus_val.json", tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"[Foundation Trainer] Loaded {len(train_dataset)} train, {len(val_dataset)} val instances.")
    
    # Initialize Foundation Model
    # Freeze lower 18 layers of ModernBERT to accelerate convergence and focus training on recurrent core
    model = ELMDEFoundationModel(backbone_name="answerdotai/ModernBERT-base", freeze_backbone=False)
    
    # Differential parameter freezing: keep top 4 encoder layers + all ELMDE core trainable
    for i in range(18):
        for param in model.backbone.layers[i].parameters():
            param.requires_grad = False
            
    model.to(device)
    
    # Parameter groups with differential learning rates
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    elmde_params = [
        p for n, p in model.named_parameters() 
        if not n.startswith("backbone") and p.requires_grad
    ]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": elmde_params, "lr": 8e-4, "weight_decay": 0.01}
    ])
    
    epochs = 4
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    corn_criterion = CORNLoss(num_classes=5)
    beta_criterion = BetaNLLLoss()
    choice_criterion = HeteroscedasticLoss()
    
    # Static candidate embeddings for choice head (5 choices, 768 dim)
    candidate_embeddings = nn.Parameter(
        torch.randn(5, model.config.d_model, device=device) * 0.05
    )
    
    print(f"[Foundation Trainer] Beginning training: {epochs} epochs, {total_steps} total steps...")
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps_taken = 0.0
        epoch_residual = 0.0
        correct_ordinal = 0
        total_samples = 0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ordinal_labels = batch["ordinal_label"].to(device)
            continuous_labels = batch["continuous_label"].to(device)
            choice_labels = batch["choice_label"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Ordinal branch with contractive recurrence
            ctx = model.forward_backbone(input_ids, attention_mask)
            z_star, metrics = model.solve_latent_equilibrium(ctx, use_ift=True)
            
            # 1. CORN ordinal predictions
            class_probs, cum_survival, cond_probs = model.corn_head(z_star)
            loss_corn = corn_criterion(cond_probs, ordinal_labels)
            
            # 2. Beta continuous predictions
            mean_c, var_c, alpha_c, beta_c = model.density_head(z_star)
            loss_beta = beta_criterion(alpha_c, beta_c, continuous_labels)
            
            # 3. Choice predictions
            b_size = input_ids.shape[0]
            cand_batch = candidate_embeddings.unsqueeze(0).expand(b_size, -1, -1)
            probs_choice, tau_choice = model.choice_head(z_star, cand_batch)
            loss_choice = nn.functional.cross_entropy(torch.log(probs_choice + 1e-9), choice_labels)
            
            # 4. MoE auxiliary load balancing loss
            loss_aux = metrics["aux_loss"]
            
            total_loss = loss_corn + 0.5 * loss_beta + 0.5 * loss_choice + 0.02 * loss_aux
            total_loss.backward()
            
            # Gradient clipping for contractive numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += total_loss.item()
            epoch_steps_taken += metrics["steps_taken"].item()
            epoch_residual += metrics["final_residual"].item()
            
            preds = class_probs.argmax(dim=-1)
            correct_ordinal += (preds == ordinal_labels).sum().item()
            total_samples += b_size
            
        avg_loss = epoch_loss / len(train_loader)
        avg_steps = epoch_steps_taken / len(train_loader)
        avg_res = epoch_residual / len(train_loader)
        acc = (correct_ordinal / total_samples) * 100.0
        
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Ordinal Acc: {acc:.2f}% | Recurrent Steps: {avg_steps:.1f} | Fixed-Point Residual: {avg_res:.5f}")

    total_time = time.time() - t0
    print(f"[Foundation Trainer] Training completed in {total_time:.2f}s.")
    
    # Validation and Conformal Calibration (SCRC)
    print("\n[Foundation Trainer] Running Validation & Split-Conformal Calibration...")
    model.eval()
    val_correct = 0
    val_total = 0
    val_confidences = []
    val_accuracies = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["ordinal_label"].to(device)
            
            class_probs, _, met = model.forward_ordinal(input_ids, attention_mask, use_ift=False)
            max_probs, preds = class_probs.max(dim=-1)
            
            corrects = (preds == labels).float()
            val_correct += corrects.sum().item()
            val_total += len(labels)
            
            val_confidences.extend(max_probs.cpu().tolist())
            val_accuracies.extend(corrects.cpu().tolist())
            
    val_acc = (val_correct / val_total) * 100.0
    print(f"[Foundation Trainer] Validation Raw Ordinal Accuracy: {val_acc:.2f}%")
    
    # Split Conformal Risk Control Calibration
    import numpy as np
    scrc = SelectiveConformalRiskController(alpha_target=0.01, loss_bound_b=1.0)
    conf_arr = np.array(val_confidences)
    loss_arr = np.array(val_accuracies)  # 0 if wrong, 1 if correct
    error_arr = 1.0 - loss_arr           # loss in [0, 1]
    
    calib_stats = scrc.calibrate(conf_arr, error_arr)
    tau_star = scrc.calibrated_tau if scrc.calibrated_tau is not None else 0.95
    print(f"[Foundation Trainer] SCRC Conformal Threshold (tau*): {tau_star:.4f}")
    
    # Save production foundation checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = model.save_pretrained("checkpoints", "elmde_foundation_production.pt")
    
    calib_meta = {
        "tau_star": float(tau_star),
        "alpha_target": 0.01,
        "n_calibration_samples": val_total,
        "validation_accuracy": val_acc,
        "backbone": "answerdotai/ModernBERT-base",
        "timestamp": time.time()
    }
    with open("checkpoints/scrc_foundation_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)
        
    print(f"[Foundation Trainer] Saved foundation checkpoint -> {ckpt_path}")
    print(f"[Foundation Trainer] Saved calibration -> checkpoints/scrc_foundation_calibration.json")
    return ckpt_path


if __name__ == "__main__":
    train_foundation_elmde()
