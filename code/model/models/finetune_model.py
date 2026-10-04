"""
Fine-tuning model with multi-level support (L0–L3).

L0: Pre-trained encoder + classification head
L1: + CoT embedding alignment head (distillation)
L2: + Metadata-aware routing mechanism
L3: + Selective expert freezing strategy
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe import MoEFeedForward, TopKRouter, Expert
from models.ts_moe_encoder import TimeSeriesMoEEncoder, TransformerMoEBlock


class MetadataAwareRouter(nn.Module):
    """
    Extended router that incorporates metadata for A-share specific routing.
    Gate input = [hidden_state; metadata_projection] to bias expert selection.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        metadata_dim: int = 8,
        jitter: float = 0.0,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.jitter = jitter

        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.gate = nn.Linear(d_model * 2, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = x.shape

        if metadata is not None:
            meta_emb = self.metadata_proj(metadata)  # [B, D]
            meta_emb = meta_emb.unsqueeze(1).expand(-1, S, -1)  # [B, S, D]
            gate_input = torch.cat([x, meta_emb], dim=-1)  # [B, S, 2D]
        else:
            gate_input = torch.cat([x, torch.zeros_like(x)], dim=-1)

        logits = self.gate(gate_input)

        if self.training and self.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.jitter

        routing_probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = routing_probs.topk(self.top_k, dim=-1)
        weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        balance_loss = self._load_balance_loss(routing_probs, top_k_indices)
        return weights, top_k_indices, balance_loss

    def _load_balance_loss(self, probs, indices):
        one_hot = F.one_hot(indices, self.num_experts).float()
        one_hot = one_hot.sum(dim=-2)
        f = one_hot.mean(dim=[0, 1])
        p = probs.mean(dim=[0, 1])
        return self.num_experts * (f * p).sum()


class FineTuneModel(nn.Module):
    """
    Multi-level fine-tuning model.
    Wraps the pre-trained encoder with task-specific heads.
    """

    def __init__(
        self,
        encoder: TimeSeriesMoEEncoder,
        n_classes: int = 2,
        level: str = "L0",
        level_cfg: dict = None,
        cot_cfg: dict = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.level = level
        self.d_model = encoder.d_model
        level_cfg = level_cfg or {}
        cot_cfg = cot_cfg or {}

        # Classification head (all levels)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_model // 2, n_classes),
        )

        # CoT distillation head (L1+)
        self.cot_proj = None
        if level in ("L1", "L2", "L3"):
            emb_dim = cot_cfg.get("embedding_dim", 384)
            proj_dim = cot_cfg.get("projection_dim", 128)
            self.cot_proj = nn.Sequential(
                nn.Linear(self.d_model, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            )
            self.cot_target_proj = nn.Sequential(
                nn.Linear(emb_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            )

        # Metadata-aware routing (L2+)
        self._metadata = None
        if level in ("L2", "L3") and level_cfg.get("custom_router"):
            self._install_metadata_routers(level_cfg.get("metadata_dim", 8))

        # Expert freezing (L3)
        if level == "L3" and level_cfg.get("expert_freeze"):
            self._freeze_n = level_cfg.get("freeze_top_n_experts", 4)
        else:
            self._freeze_n = 0

    def _install_metadata_routers(self, metadata_dim: int):
        """Replace standard routers with metadata-aware versions."""
        for block in self.encoder.blocks:
            old_moe = block.moe_ffn
            new_router = MetadataAwareRouter(
                d_model=self.d_model,
                num_experts=old_moe.num_routed,
                top_k=old_moe.top_k,
                metadata_dim=metadata_dim,
            )
            old_moe.router = new_router
            old_moe._metadata_router = True

    def freeze_experts(self, expert_activation_ranks: list[int]):
        """
        Freeze the top-N most activated experts (universal experts).
        expert_activation_ranks: list of expert indices sorted by activation frequency.
        """
        if self._freeze_n <= 0:
            return
        experts_to_freeze = expert_activation_ranks[:self._freeze_n]
        for block in self.encoder.blocks:
            for e_id in experts_to_freeze:
                expert = block.moe_ffn.routed_experts[e_id]
                for p in expert.parameters():
                    p.requires_grad = False

    def set_metadata(self, metadata: torch.Tensor):
        """Set batch-level metadata for metadata-aware routing."""
        self._metadata = metadata

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = batch["features"]
        padding_mask = batch.get("padding_mask")

        metadata = batch.get("metadata")
        if metadata is None and hasattr(self, '_metadata') and self._metadata is not None:
            metadata = self._metadata

        seq_out, cls_out, moe_loss = self.encoder(features, mask=None, metadata=metadata)

        logits = self.classifier(cls_out)

        result = {
            "logits": logits,
            "cls_embedding": cls_out,
            "moe_loss": moe_loss,
        }

        if self.cot_proj is not None and "cot_embedding" in batch:
            cls_proj = self.cot_proj(cls_out)
            cls_proj = F.normalize(cls_proj, dim=-1)

            cot_target = self.cot_target_proj(batch["cot_embedding"])
            cot_target = F.normalize(cot_target, dim=-1)

            result["cls_proj"] = cls_proj
            result["cot_target_proj"] = cot_target

        return result

    def get_expert_activations(
        self, dataloader, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Collect expert activation statistics across a dataset."""
        self.eval()
        n_layers = len(self.encoder.blocks)
        n_experts = self.encoder.blocks[0].moe_ffn.num_routed
        counts = torch.zeros(n_layers, n_experts)

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(device)
                B, S, _ = features.shape
                x = self.encoder.input_proj(features)
                x = self.encoder.input_norm(x)
                x = x + self.encoder.pos_embedding[:, :S, :]

                cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

                for layer_idx, block in enumerate(self.encoder.blocks):
                    residual = x
                    x_norm = block.norm1(x)
                    x_attn, _ = block.attn(x_norm, x_norm, x_norm)
                    x = block.dropout(x_attn) + residual

                    residual = x
                    x_norm = block.norm2(x)
                    _, indices, _ = block.moe_ffn.router(x_norm)
                    for e_id in range(n_experts):
                        counts[layer_idx, e_id] += (indices == e_id).float().sum().cpu()

                    ffn_out, _ = block.moe_ffn(x_norm)
                    x = block.dropout(ffn_out) + residual

                x = self.encoder.final_norm(x)

        total = counts.sum(dim=1, keepdim=True)
        utilization = counts / (total + 1e-9)
        avg_activation = utilization.mean(dim=0)
        expert_ranks = avg_activation.argsort(descending=True).tolist()

        return {
            "counts": counts,
            "utilization": utilization,
            "avg_activation": avg_activation,
            "expert_ranks": expert_ranks,
        }

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        model_cfg: dict,
        n_features: int,
        level: str = "L0",
        level_cfg: dict = None,
        cot_cfg: dict = None,
    ) -> "FineTuneModel":
        """Load pre-trained encoder and build fine-tuning model."""
        encoder = TimeSeriesMoEEncoder(
            n_input_features=n_features,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            max_seq_len=model_cfg["max_seq_len"],
            moe_cfg=model_cfg["moe"],
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", {}))

            if "encoder_state_dict" in ckpt:
                encoder.load_state_dict(state_dict, strict=False)
            else:
                filtered = {
                    k.replace("encoder.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("encoder.")
                }
                if filtered:
                    encoder.load_state_dict(filtered, strict=False)

            from loguru import logger
            logger.info(f"Loaded pre-trained encoder from {checkpoint_path}")

        model = cls(
            encoder=encoder,
            level=level,
            level_cfg=level_cfg,
            cot_cfg=cot_cfg,
        )
        return model
