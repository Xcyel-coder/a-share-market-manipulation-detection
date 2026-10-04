"""
Fine-tuning trainer with multi-level support, Focal Loss, and K-Fold CV.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from loguru import logger


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in manipulation detection."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FineTuneTrainer:
    """Handles fine-tuning with multi-level support."""

    def __init__(self, cfg: dict, output_dir: str):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = cfg["training"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["lr"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.patience = train_cfg["early_stopping"]["patience"]
        self.min_delta = train_cfg["early_stopping"]["min_delta"]

        focal_cfg = train_cfg["focal_loss"]
        self.cls_loss_fn = FocalLoss(gamma=focal_cfg["gamma"], alpha=focal_cfg["alpha"])

        self.moe_balance_weight = cfg["model"]["moe"]["load_balance_weight"]

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        level: str,
        loss_weights: dict,
        fold: int = 0,
    ) -> dict:
        """Run fine-tuning for one fold."""
        run_dir = self.output_dir / f"{level}" / f"fold_{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)

        model = model.to(self.device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.cfg["training"]["weight_decay"])

        total_steps = self.epochs * len(train_loader)
        warmup_steps = int(total_steps * self.cfg["training"]["warmup_ratio"])

        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_steps, 1))
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
        scheduler = SequentialLR(
            optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps]
        )

        best_f1 = 0.0
        best_metrics = {}
        no_improve = 0
        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}

        w_cls = loss_weights.get("classification", 1.0)
        w_cot = loss_weights.get("cot_distill", 0.0)

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(
                model, train_loader, optimizer, scheduler, w_cls, w_cot
            )
            val_metrics = self._validate(model, val_loader, w_cls, w_cot)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_auc"].append(val_metrics["auc"])

            logger.info(
                f"[{level}/fold{fold}] Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | "
                f"P: {val_metrics['precision']:.4f} R: {val_metrics['recall']:.4f}"
            )

            if val_metrics["f1"] > best_f1 + self.min_delta:
                best_f1 = val_metrics["f1"]
                best_metrics = val_metrics.copy()
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": val_metrics,
                    "level": level,
                    "fold": fold,
                }, run_dir / "best.pt")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        history["best_metrics"] = best_metrics
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        return history

    def _train_epoch(
        self, model, loader, optimizer, scheduler, w_cls, w_cot,
    ) -> float:
        model.train()
        total_loss = 0
        n = 0

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(batch)
            loss = self._compute_loss(outputs, batch, w_cls, w_cot)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _validate(self, model, loader, w_cls, w_cot) -> dict:
        model.eval()
        all_logits, all_labels = [], []
        total_loss = 0
        n = 0

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(batch)
            loss = self._compute_loss(outputs, batch, w_cls, w_cot)
            total_loss += loss.item()
            n += 1

            all_logits.append(outputs["logits"].cpu())
            all_labels.append(batch["label"].cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        return self._compute_metrics(logits, labels, total_loss / max(n, 1))

    def _compute_loss(
        self, outputs, batch, w_cls, w_cot,
    ) -> torch.Tensor:
        cls_loss = self.cls_loss_fn(outputs["logits"], batch["label"])
        moe_loss = outputs["moe_loss"]
        total = w_cls * cls_loss + self.moe_balance_weight * moe_loss

        if w_cot > 0 and "cls_proj" in outputs and "cot_target_proj" in outputs:
            has_cot = batch.get("has_cot", torch.ones(outputs["logits"].shape[0], device=cls_loss.device))
            if has_cot.sum() > 0:
                cos_sim = F.cosine_similarity(
                    outputs["cls_proj"], outputs["cot_target_proj"], dim=-1
                )
                cot_loss = (1 - cos_sim) * has_cot
                cot_loss = cot_loss.sum() / (has_cot.sum() + 1e-9)
                total = total + w_cot * cot_loss

        return total

    def _compute_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor, loss: float,
    ) -> dict:
        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = (probs >= 0.5).long()

        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()

        precision = (tp / (tp + fp + 1e-9)).item()
        recall = (tp / (tp + fn + 1e-9)).item()
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy = ((tp + tn) / (tp + fp + fn + tn + 1e-9)).item()

        auc = self._compute_auc(probs, labels)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }

    def _compute_auc(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Simple AUC computation without sklearn dependency."""
        if labels.sum() == 0 or labels.sum() == len(labels):
            return 0.5

        pos = probs[labels == 1]
        neg = probs[labels == 0]
        n_pos, n_neg = len(pos), len(neg)

        correct = 0
        for p in pos:
            correct += (p > neg).sum().item()
            correct += 0.5 * (p == neg).sum().item()

        return correct / (n_pos * n_neg + 1e-9)
