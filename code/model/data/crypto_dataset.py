"""
Crypto pre-training dataset.

Loads processed crypto K-line data and creates sliding windows for:
1. Masked Autoencoder (MAE): reconstruct masked timesteps
2. Anomaly prediction: binary classification of future anomaly
3. Contrastive learning: positive/negative pair sampling
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data.feature_engineering import (
    compute_features,
    detect_anomalies,
    normalize_window,
    get_feature_names,
)


class CryptoPretrainDataset(Dataset):
    """
    Sliding-window dataset for self-supervised pre-training.

    Each sample is a dict:
        features:     [seq_len, n_features]  — normalized feature window
        mask:         [seq_len]              — 1 = masked, 0 = visible
        targets:      [seq_len, n_features]  — original values (for MAE loss)
        anomaly_label: scalar                — 1 if lookahead contains anomaly
        is_anomaly_window: scalar            — 1 if window overlaps anomaly region
    """

    def __init__(
        self,
        data_dir: str,
        symbols: list[str],
        seq_len: int = 256,
        stride: int = 64,
        mask_ratio: float = 0.25,
        block_mask: bool = True,
        min_block_len: int = 4,
        max_block_len: int = 16,
        lookahead_window: int = 10,
        anomaly_cfg: Optional[dict] = None,
        data_fraction: float = 1.0,
        is_validation: bool = False,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.block_mask = block_mask
        self.min_block_len = min_block_len
        self.max_block_len = max_block_len
        self.lookahead_window = lookahead_window

        anomaly_cfg = anomaly_cfg or {}

        all_windows = []
        all_anomaly_labels = []
        all_is_anomaly = []

        for symbol in symbols:
            symbol_dir = Path(data_dir) / symbol
            if not symbol_dir.exists():
                continue

            df = self._load_symbol(symbol_dir)
            if df is None or len(df) < seq_len + lookahead_window:
                continue

            df = compute_features(df)
            anomaly_flags = detect_anomalies(df, **anomaly_cfg)

            feature_cols = get_feature_names()
            feat_matrix = df[feature_cols].values.astype(np.float32)

            # Replace inf/nan
            feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=3.0, neginf=-3.0)

            # Time-based split
            n = len(feat_matrix)
            split_idx = int(n * (1 - val_ratio))

            if is_validation:
                feat_matrix = feat_matrix[split_idx:]
                anomaly_flags = anomaly_flags[split_idx:]
            else:
                feat_matrix = feat_matrix[:split_idx]
                anomaly_flags = anomaly_flags[:split_idx]

            # Create sliding windows
            n = len(feat_matrix)
            for start in range(0, n - seq_len - lookahead_window, stride):
                end = start + seq_len
                window = feat_matrix[start:end]
                lookahead = anomaly_flags[end:end + lookahead_window]

                all_windows.append(window)
                all_anomaly_labels.append(int(lookahead.any()))
                all_is_anomaly.append(int(anomaly_flags[start:end].any()))

        self.windows = np.array(all_windows, dtype=np.float32)
        self.anomaly_labels = np.array(all_anomaly_labels, dtype=np.int64)
        self.is_anomaly = np.array(all_is_anomaly, dtype=np.int64)
        self.n_features = self.windows.shape[2] if len(self.windows) > 0 else 0

        # Sub-sample for progressive training
        if data_fraction < 1.0 and len(self.windows) > 0:
            n_samples = max(1, int(len(self.windows) * data_fraction))
            rng = np.random.RandomState(42)
            indices = rng.choice(len(self.windows), n_samples, replace=False)
            indices.sort()
            self.windows = self.windows[indices]
            self.anomaly_labels = self.anomaly_labels[indices]
            self.is_anomaly = self.is_anomaly[indices]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = self.windows[idx].copy()

        # Normalize within window (instance normalization)
        targets = window.copy()
        window = normalize_window(window)

        # Generate mask
        mask = self._generate_mask()

        return {
            "features": torch.from_numpy(window),
            "targets": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask),
            "anomaly_label": torch.tensor(self.anomaly_labels[idx], dtype=torch.long),
            "is_anomaly_window": torch.tensor(self.is_anomaly[idx], dtype=torch.long),
        }

    def _generate_mask(self) -> np.ndarray:
        """Generate masking pattern for MAE."""
        mask = np.zeros(self.seq_len, dtype=np.float32)
        n_mask = int(self.seq_len * self.mask_ratio)

        if self.block_mask:
            masked = 0
            while masked < n_mask:
                block_len = np.random.randint(self.min_block_len, self.max_block_len + 1)
                block_len = min(block_len, n_mask - masked)
                start = np.random.randint(0, self.seq_len - block_len + 1)
                mask[start:start + block_len] = 1.0
                masked = int(mask.sum())
        else:
            indices = np.random.choice(self.seq_len, n_mask, replace=False)
            mask[indices] = 1.0

        return mask

    def _load_symbol(self, symbol_dir: Path) -> Optional[pd.DataFrame]:
        """Load and concatenate all parquet shards for a symbol."""
        parquet_files = sorted(symbol_dir.glob("*.parquet"))
        if not parquet_files:
            return None
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_anomaly_ratio(self) -> float:
        """Return the fraction of anomaly windows (for class weight)."""
        if len(self.is_anomaly) == 0:
            return 0.0
        return self.is_anomaly.mean()

    def get_contrastive_pairs(
        self, batch_indices: list[int], n_negatives: int = 4
    ) -> tuple[list[int], list[int]]:
        """
        For contrastive learning: given a batch, return indices of
        positive (anomaly) and negative (normal) windows.
        """
        positives = [i for i in batch_indices if self.is_anomaly[i]]
        negatives = [i for i in batch_indices if not self.is_anomaly[i]]
        return positives, negatives


def build_dataloaders(
    cfg: dict,
    round_cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders from config."""
    data_cfg = cfg["data"]
    pretrain_cfg = cfg["pretrain"]
    mae_cfg = pretrain_cfg["mae"]
    anomaly_cfg = data_cfg.get("anomaly", {})

    common_kwargs = dict(
        data_dir=data_cfg["raw_dir"],
        symbols=data_cfg["symbols"],
        seq_len=data_cfg["seq_len"],
        stride=data_cfg["stride"],
        mask_ratio=mae_cfg["mask_ratio"],
        block_mask=mae_cfg["block_mask"],
        min_block_len=mae_cfg["min_block_len"],
        max_block_len=mae_cfg["max_block_len"],
        lookahead_window=pretrain_cfg["anomaly_pred"]["lookahead_window"],
        anomaly_cfg={
            "price_surge_threshold": anomaly_cfg.get("price_surge_threshold", 0.05),
            "volume_multiplier": anomaly_cfg.get("volume_multiplier", 3.0),
            "surge_window": anomaly_cfg.get("surge_window", 5),
            "min_gap": anomaly_cfg.get("min_gap_between", 30),
        },
        val_ratio=data_cfg.get("val_ratio", 0.1),
    )

    train_ds = CryptoPretrainDataset(
        **common_kwargs,
        data_fraction=round_cfg["data_fraction"],
        is_validation=False,
    )
    val_ds = CryptoPretrainDataset(
        **common_kwargs,
        data_fraction=1.0,
        is_validation=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=round_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=round_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader
