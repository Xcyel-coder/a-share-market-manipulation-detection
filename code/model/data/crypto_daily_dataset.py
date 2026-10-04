"""
Daily/4h-scale crypto pre-training dataset for scale-aligned transfer.

Uses the SAME 25-feature pipeline as A-stock fine-tuning (compute_daily_features),
ensuring the encoder's feature space is fully aligned between pre-training and
downstream fine-tuning.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader

from data.astock_dataset import compute_daily_features
from data.feature_engineering import normalize_window


def detect_daily_anomalies(
    df: pd.DataFrame,
    price_surge_threshold: float = 0.05,
    volume_multiplier: float = 2.0,
    surge_window: int = 3,
    min_gap: int = 5,
    volume_ma_window: int = 20,
) -> np.ndarray:
    """
    Detect statistical anomalies in daily/4h-aggregated crypto data.
    Uses adapted thresholds for daily-scale price moves.
    """
    n = len(df)
    labels = np.zeros(n, dtype=np.int64)
    c = df["close"].values
    v = df["volume"].values
    vol_ma = pd.Series(v).rolling(volume_ma_window, min_periods=5).mean().values

    last_anomaly = -min_gap
    for i in range(surge_window, n - surge_window):
        window_end = min(i + surge_window, n - 1)
        max_price = c[i:window_end + 1].max()
        surge = (max_price - c[i]) / c[i] if c[i] > 0 else 0
        vol_ratio = v[i] / vol_ma[i] if vol_ma[i] and vol_ma[i] > 0 else 0
        if surge >= price_surge_threshold and vol_ratio >= volume_multiplier:
            if i - last_anomaly >= min_gap:
                labels[i:window_end + 1] = 1
                last_anomaly = window_end
    return labels


class CryptoDailyPretrainDataset(Dataset):
    """
    Sliding-window dataset for daily/4h crypto pre-training.

    Key difference from CryptoPretrainDataset: uses compute_daily_features()
    (the A-stock pipeline) to produce exactly 25 features that match the
    fine-tuning feature space, enabling scale-aligned transfer.
    """

    def __init__(
        self,
        data_dir: str,
        symbols: list[str],
        timeframe: str = "daily",
        seq_len: int = 60,
        stride: int = 10,
        mask_ratio: float = 0.25,
        block_mask: bool = True,
        min_block_len: int = 2,
        max_block_len: int = 8,
        lookahead_window: int = 5,
        anomaly_cfg: Optional[dict] = None,
        feature_cfg: Optional[dict] = None,
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
        self.timeframe = timeframe

        anomaly_cfg = anomaly_cfg or {}
        feature_cfg = feature_cfg or {}

        all_windows = []
        all_anomaly_labels = []
        all_is_anomaly = []
        self.feature_names = []

        for symbol in symbols:
            fname = f"{symbol}_{timeframe}.parquet"
            fpath = Path(data_dir) / fname
            if not fpath.exists():
                continue

            df = pd.read_parquet(fpath)
            if "datetime" in df.columns:
                df["date"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("date").reset_index(drop=True)

            if len(df) < seq_len + lookahead_window + 60:
                continue

            df, feat_names = compute_daily_features(df, **feature_cfg)
            self.feature_names = feat_names
            anomaly_flags = detect_daily_anomalies(df, **anomaly_cfg)

            feat_matrix = df[feat_names].values.astype(np.float32)
            feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=3.0, neginf=-3.0)

            n = len(feat_matrix)
            split_idx = int(n * (1 - val_ratio))

            if is_validation:
                feat_matrix = feat_matrix[split_idx:]
                anomaly_flags = anomaly_flags[split_idx:]
            else:
                feat_matrix = feat_matrix[:split_idx]
                anomaly_flags = anomaly_flags[:split_idx]

            n = len(feat_matrix)
            for start in range(0, n - seq_len - lookahead_window, stride):
                end = start + seq_len
                window = feat_matrix[start:end]
                lookahead = anomaly_flags[end:end + lookahead_window]
                all_windows.append(window)
                all_anomaly_labels.append(int(lookahead.any()))
                all_is_anomaly.append(int(anomaly_flags[start:end].any()))

        if all_windows:
            self.windows = np.array(all_windows, dtype=np.float32)
        else:
            self.windows = np.zeros((0, seq_len, 25), dtype=np.float32)
        self.anomaly_labels = np.array(all_anomaly_labels, dtype=np.int64)
        self.is_anomaly = np.array(all_is_anomaly, dtype=np.int64)
        self.n_features = self.windows.shape[2] if len(self.windows) > 0 else 25

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
        targets = window.copy()
        window = normalize_window(window)
        mask = self._generate_mask()

        return {
            "features": torch.from_numpy(window),
            "targets": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask),
            "anomaly_label": torch.tensor(self.anomaly_labels[idx], dtype=torch.long),
            "is_anomaly_window": torch.tensor(self.is_anomaly[idx], dtype=torch.long),
        }

    def _generate_mask(self) -> np.ndarray:
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

    def get_anomaly_ratio(self) -> float:
        if len(self.is_anomaly) == 0:
            return 0.0
        return self.is_anomaly.mean()


def build_daily_dataloaders(
    cfg: dict,
    round_cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders for daily/4h pre-training."""
    data_cfg = cfg["data"]
    pretrain_cfg = cfg["pretrain"]
    mae_cfg = pretrain_cfg["mae"]
    anomaly_cfg = data_cfg.get("anomaly", {})
    feature_cfg = data_cfg.get("feature_params", {})

    common_kwargs = dict(
        data_dir=data_cfg["aggregated_dir"],
        symbols=data_cfg["symbols"],
        timeframe=round_cfg.get("timeframe", "daily"),
        seq_len=data_cfg["seq_len"],
        stride=data_cfg["stride"],
        mask_ratio=mae_cfg["mask_ratio"],
        block_mask=mae_cfg["block_mask"],
        min_block_len=mae_cfg["min_block_len"],
        max_block_len=mae_cfg["max_block_len"],
        lookahead_window=pretrain_cfg["anomaly_pred"]["lookahead_window"],
        anomaly_cfg={
            "price_surge_threshold": anomaly_cfg.get("price_surge_threshold", 0.05),
            "volume_multiplier": anomaly_cfg.get("volume_multiplier", 2.0),
            "surge_window": anomaly_cfg.get("surge_window", 3),
            "min_gap": anomaly_cfg.get("min_gap", 5),
            "volume_ma_window": anomaly_cfg.get("volume_ma_window", 20),
        },
        feature_cfg=feature_cfg,
        val_ratio=data_cfg.get("val_ratio", 0.1),
    )

    train_ds = CryptoDailyPretrainDataset(
        **common_kwargs,
        data_fraction=round_cfg.get("data_fraction", 1.0),
        is_validation=False,
    )
    val_ds = CryptoDailyPretrainDataset(
        **common_kwargs,
        data_fraction=1.0,
        is_validation=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=round_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=round_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
