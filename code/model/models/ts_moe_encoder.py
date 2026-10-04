"""
Time-Series MoE Encoder with pre-training task heads.

Architecture:
    Input [B, S, F] → Projection → Positional Encoding
    → N × TransformerMoEBlock (Self-Attention + MoE FFN)
    → Encoder output [B, S, D]

Pre-training heads:
    1. MAE head:         reconstruct masked timesteps → [B, S, F]
    2. Anomaly head:     predict future anomaly       → [B, 1]
    3. Contrastive head: project to embedding space   → [B, D_proj]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe import MoEFeedForward


class TransformerMoEBlock(nn.Module):
    """Single Transformer block with MoE feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        moe_cfg: dict,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.moe_ffn = MoEFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            num_routed_experts=moe_cfg["num_routed_experts"],
            num_shared_experts=moe_cfg["num_shared_experts"],
            top_k=moe_cfg["top_k"],
            router_jitter=moe_cfg.get("router_jitter", 0.1),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        metadata: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, D]
            attn_mask: optional attention mask
            metadata: optional [B, metadata_dim] for metadata-aware routing
        Returns:
            output: [B, S, D]
            moe_loss: scalar load balancing loss
        """
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        ffn_out, moe_loss = self.moe_ffn(x, metadata=metadata)
        x = self.dropout(ffn_out) + residual

        return x, moe_loss


class TimeSeriesMoEEncoder(nn.Module):
    """
    Transformer encoder with MoE layers for financial time series.

    This is the core model that:
    - Pre-trains on crypto data (Phase 2)
    - Gets fine-tuned on A-share data (Phase 3)
    - Connects to Qwen for CoT generation (Phase 3)
    """

    def __init__(
        self,
        n_input_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        moe_cfg: dict = None,
    ):
        super().__init__()
        self.d_model = d_model

        moe_cfg = moe_cfg or {
            "num_routed_experts": 8,
            "num_shared_experts": 2,
            "top_k": 2,
            "router_jitter": 0.1,
        }

        # Input projection
        self.input_proj = nn.Linear(n_input_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # CLS token for sequence-level tasks
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(d_model, n_heads, d_ff, dropout, moe_cfg)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        metadata: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, n_features] — input features
            mask: [B, S] — 1 = masked position (for MAE)
            metadata: optional [B, metadata_dim] for metadata-aware routing

        Returns:
            seq_output:  [B, S, d_model]  — per-timestep representations
            cls_output:  [B, d_model]     — sequence-level representation
            total_moe_loss: scalar        — sum of load balancing losses
        """
        B, S, _ = x.shape

        x = self.input_proj(x)
        x = self.input_norm(x)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x = x * (1 - mask_expanded)

        x = x + self.pos_embedding[:, :S, :]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        total_moe_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, moe_loss = block(x, metadata=metadata)
            total_moe_loss = total_moe_loss + moe_loss

        x = self.final_norm(x)

        cls_output = x[:, 0, :]
        seq_output = x[:, 1:, :]

        return seq_output, cls_output, total_moe_loss


class PreTrainingModel(nn.Module):
    """
    Complete pre-training model = Encoder + 3 task heads.

    Task 1 (MAE):         Reconstruct masked timesteps
    Task 2 (Anomaly):     Predict whether future window has anomaly
    Task 3 (Contrastive): Project to contrastive embedding space
    """

    def __init__(self, encoder: TimeSeriesMoEEncoder, n_features: int):
        super().__init__()
        self.encoder = encoder
        d = encoder.d_model

        # Head 1: Masked reconstruction
        self.mae_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, n_features),
        )

        # Head 2: Anomaly prediction (binary)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, 2),
        )

        # Head 3: Contrastive projection
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 128),
        )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys 'features', 'mask', 'targets',
                   'anomaly_label', 'is_anomaly_window'
        Returns:
            dict of outputs for each task head
        """
        features = batch["features"]    # [B, S, F]
        mask = batch["mask"]            # [B, S]

        seq_out, cls_out, moe_loss = self.encoder(features, mask)

        # Task 1: MAE reconstruction
        mae_pred = self.mae_head(seq_out)  # [B, S, F]

        # Task 2: Anomaly prediction
        anomaly_logits = self.anomaly_head(cls_out)  # [B, 2]

        # Task 3: Contrastive embedding
        contrast_emb = self.contrastive_proj(cls_out)  # [B, 128]
        contrast_emb = F.normalize(contrast_emb, dim=-1)

        return {
            "mae_pred": mae_pred,
            "anomaly_logits": anomaly_logits,
            "contrast_emb": contrast_emb,
            "moe_loss": moe_loss,
        }

    @classmethod
    def from_config(cls, model_cfg: dict, n_features: int) -> "PreTrainingModel":
        """Build model from YAML config dict."""
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
        return cls(encoder, n_features)
