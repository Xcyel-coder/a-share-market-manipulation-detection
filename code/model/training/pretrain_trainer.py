"""
Pre-training trainer with 3-round progressive strategy.

Round 1: 10% data, validate training pipeline works
Round 2: 50% data, tune hyperparameters
Round 3: 100% data, full training for final checkpoint
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from loguru import logger


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with temperature scaling."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings
            labels: [B] binary labels (1=anomaly, 0=normal)
        """
        B = embeddings.shape[0]
        if B < 2 or labels.sum() == 0 or labels.sum() == B:
            return torch.tensor(0.0, device=embeddings.device)

        sim = torch.mm(embeddings, embeddings.t()) / self.temperature
        # mask out self-similarity
        mask_self = ~torch.eye(B, dtype=torch.bool, device=sim.device)

        # positive pairs: same label
        label_match = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = label_match & mask_self
        neg_mask = ~label_match & mask_self

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # for each anchor, compute loss over its positive pairs
        exp_sim = torch.exp(sim) * mask_self.float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        n_pos = pos_mask.float().sum(dim=1).clamp(min=1)

        loss = -(pos_log_prob / n_pos).mean()
        return loss


class PreTrainTrainer:
    """Handles the 3-round progressive pre-training."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        output_dir: str,
    ):
        self.model = model
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg["training"]
        pretrain_cfg = cfg["pretrain"]
        self.loss_weights = pretrain_cfg["loss_weights"]
        self.moe_balance_weight = cfg["model"]["moe"]["load_balance_weight"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.use_fp16 = train_cfg.get("fp16", True)
        self.log_every = train_cfg.get("log_every_n_steps", 50)
        self.eval_every = train_cfg.get("eval_every_n_steps", 500)
        self.save_every_epochs = train_cfg.get("save_every_n_epochs", 5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scaler = GradScaler(enabled=self.use_fp16)

        self.contrastive_loss_fn = InfoNCELoss(
            pretrain_cfg["contrastive"]["temperature"]
        )

    def train_round(
        self,
        round_idx: int,
        round_cfg: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """Execute one training round."""
        round_name = round_cfg["name"]
        epochs = round_cfg["epochs"]
        lr = round_cfg["lr"]
        warmup_ratio = round_cfg["warmup_ratio"]

        round_dir = self.output_dir / round_name
        round_dir.mkdir(exist_ok=True)

        logger.info(f"{'='*60}")
        logger.info(f"Starting {round_name}: epochs={epochs}, lr={lr}, "
                     f"data={len(train_loader.dataset)} samples")
        logger.info(f"{'='*60}")

        optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.cfg["training"]["weight_decay"],
        )

        total_steps = epochs * len(train_loader)
        warmup_steps = int(total_steps * warmup_ratio)

        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(
            optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps]
        )

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "lr": []}
        global_step = 0

        for epoch in range(epochs):
            train_metrics = self._train_epoch(
                train_loader, optimizer, scheduler, epoch, global_step
            )
            global_step += len(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["lr"].append(optimizer.param_groups[0]["lr"])

            val_metrics = self._validate(val_loader)
            history["val_loss"].append(val_metrics["loss"])

            logger.info(
                f"[{round_name}] Epoch {epoch+1}/{epochs} | "
                f"Train: {train_metrics['loss']:.4f} | "
                f"Val: {val_metrics['loss']:.4f} | "
                f"MAE: {val_metrics['mae']:.4f} | "
                f"Anomaly Acc: {val_metrics['anomaly_acc']:.3f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save_checkpoint(round_dir / "best.pt", epoch, val_metrics)

            if (epoch + 1) % self.save_every_epochs == 0:
                self._save_checkpoint(
                    round_dir / f"epoch_{epoch+1}.pt", epoch, val_metrics
                )

        self._save_checkpoint(round_dir / "final.pt", epochs - 1, val_metrics)

        logger.info(f"Round {round_name} complete. Best val loss: {best_val_loss:.4f}")
        return history

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        global_step_start: int,
    ) -> dict:
        self.model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(enabled=self.use_fp16):
                outputs = self.model(batch)
                loss, loss_components = self._compute_loss(outputs, batch)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            global_step = global_step_start + step
            if (step + 1) % self.log_every == 0:
                elapsed = time.time() - t0
                speed = (step + 1) / elapsed
                logger.info(
                    f"  Step {step+1}/{len(loader)} | "
                    f"Loss: {loss.item():.4f} "
                    f"(MAE:{loss_components['mae']:.3f} "
                    f"Anom:{loss_components['anomaly']:.3f} "
                    f"Contr:{loss_components['contrastive']:.3f} "
                    f"MoE:{loss_components['moe']:.4f}) | "
                    f"{speed:.1f} it/s"
                )

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0
        total_mae = 0
        correct = 0
        total = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast(enabled=self.use_fp16):
                outputs = self.model(batch)
                loss, components = self._compute_loss(outputs, batch)

            total_loss += loss.item()
            total_mae += components["mae"]

            preds = outputs["anomaly_logits"].argmax(dim=-1)
            correct += (preds == batch["anomaly_label"]).sum().item()
            total += preds.shape[0]

        n = max(len(loader), 1)
        return {
            "loss": total_loss / n,
            "mae": total_mae / n,
            "anomaly_acc": correct / max(total, 1),
        }

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted sum of all pre-training losses."""
        mask = batch["mask"]  # [B, S]
        w = self.loss_weights

        # 1) MAE loss: only on masked positions
        mae_pred = outputs["mae_pred"]      # [B, S, F]
        targets = batch["targets"]          # [B, S, F]
        mask_3d = mask.unsqueeze(-1)        # [B, S, 1]
        mae_loss = ((mae_pred - targets) ** 2 * mask_3d).sum() / (
            mask_3d.sum() * targets.shape[-1] + 1e-9
        )

        # 2) Anomaly prediction loss
        anomaly_loss = F.cross_entropy(
            outputs["anomaly_logits"], batch["anomaly_label"]
        )

        # 3) Contrastive loss
        contrastive_loss = self.contrastive_loss_fn(
            outputs["contrast_emb"], batch["is_anomaly_window"]
        )

        # 4) MoE load balancing
        moe_loss = outputs["moe_loss"]

        total = (
            w["mae"] * mae_loss
            + w["anomaly"] * anomaly_loss
            + w["contrastive"] * contrastive_loss
            + self.moe_balance_weight * moe_loss
        )

        components = {
            "mae": mae_loss.item(),
            "anomaly": anomaly_loss.item(),
            "contrastive": contrastive_loss.item() if contrastive_loss.item() == contrastive_loss.item() else 0.0,
            "moe": moe_loss.item(),
        }

        return total, components

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "encoder_state_dict": self.model.encoder.state_dict(),
            "metrics": metrics,
        }, path)
        logger.info(f"Checkpoint saved: {path}")
