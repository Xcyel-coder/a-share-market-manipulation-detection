"""
A-share manipulation detection dataset for fine-tuning.

Loads daily OHLCV data for each case, applies feature engineering,
creates sliding windows, and assigns manipulation labels based on
overlap with known manipulation periods.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.feature_engineering import normalize_window


DAILY_FEATURE_NAMES: list[str] = []


def compute_daily_features(
    df: pd.DataFrame,
    returns_horizons: list[int] = (1, 5, 10, 20, 60),
    volume_ma_window: int = 20,
    rsi_period: int = 14,
    bb_period: int = 20,
    atr_period: int = 14,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Feature engineering for daily A-share data.
    Produces 25 features (matching crypto pre-training dimension).
    """
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].replace(0, np.nan)

    features = {}

    # --- Price-based (5) ---
    features["log_return"] = np.log(c / c.shift(1))
    features["high_low_pct"] = (h - l) / c
    features["close_open_pct"] = (c - o) / o.replace(0, np.nan)
    features["upper_shadow_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    features["lower_shadow_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c

    # --- Multi-horizon returns (5) ---
    for horizon in returns_horizons:
        features[f"return_{horizon}d"] = c.pct_change(horizon)

    # --- Volume features (3) ---
    vol_ma = v.rolling(volume_ma_window, min_periods=5).mean()
    vol_std = v.rolling(volume_ma_window, min_periods=5).std()
    features["volume_ma_ratio"] = v / vol_ma.replace(0, np.nan)
    features["volume_zscore"] = (v - vol_ma) / vol_std.replace(0, np.nan)
    features["volume_log_change"] = np.log(v / v.shift(1).replace(0, np.nan))

    # --- Turnover & VWAP proxy (2) ---
    if "turn" in df.columns:
        turn = df["turn"].astype(float)
        turn_ma = turn.rolling(volume_ma_window, min_periods=5).mean()
        features["turnover_ratio"] = turn / turn_ma.replace(0, np.nan)
    else:
        features["turnover_ratio"] = pd.Series(1.0, index=df.index)

    if "amount" in df.columns:
        vwap = df["amount"].astype(float) / v.replace(0, np.nan)
        features["vwap_deviation"] = (c - vwap) / c
    else:
        features["vwap_deviation"] = pd.Series(0.0, index=df.index)

    # --- Volatility (2) ---
    log_ret = features["log_return"]
    features["volatility_10d"] = log_ret.rolling(10, min_periods=3).std()
    features["volatility_20d"] = log_ret.rolling(20, min_periods=5).std()

    # --- Technical indicators (4) ---
    features["rsi"] = _compute_rsi(c, rsi_period)

    bb_ma = c.rolling(bb_period, min_periods=5).mean()
    bb_std = c.rolling(bb_period, min_periods=5).std()
    bb_range = (4 * bb_std).replace(0, np.nan)
    features["bb_position"] = (c - (bb_ma - 2 * bb_std)) / bb_range

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features["macd_hist_norm"] = (macd_line - signal_line) / c

    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=5).mean()
    features["atr_ratio"] = atr / c

    # --- Time features (4) ---
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
    else:
        dt = pd.to_datetime(df.index)
    dow = dt.dt.dayofweek.astype(float)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    month = dt.dt.month.astype(float)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)

    feature_names = list(features.keys())

    for name, series in features.items():
        df[name] = series

    global DAILY_FEATURE_NAMES
    DAILY_FEATURE_NAMES = feature_names

    return df, feature_names


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi / 100.0


def load_case_data(
    case_info: dict,
    astock_dir: str,
) -> dict | None:
    """Load daily data and metadata for a single case."""
    doc_id = case_info["doc_id"]
    case_dir = Path(astock_dir) / doc_id

    parquet_path = case_dir / "daily.parquet"
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)
    if len(df) < 20:
        return None

    start = case_info.get("manipulation_start_date")
    end = case_info.get("manipulation_end_date")
    if not start or not end:
        return None

    return {
        "doc_id": doc_id,
        "df": df,
        "manipulation_start": pd.to_datetime(start),
        "manipulation_end": pd.to_datetime(end),
        "methods": case_info.get("manipulation_methods", []),
        "stock_code": case_info.get("stock_code", ""),
    }


def _build_metadata(stock_code: str, window_features: np.ndarray,
                     window_dates: pd.DatetimeIndex) -> np.ndarray:
    """Build 8-dim metadata vector from available case information.

    Dims: [is_sse, is_szse_main, is_chinext, has_limit_move,
           window_volatility, window_trend, avg_volume_ratio, turnover_intensity]
    """
    meta = np.zeros(8, dtype=np.float32)
    code = str(stock_code)
    if code.startswith("6"):
        meta[0] = 1.0  # SSE main board
    elif code.startswith("0"):
        meta[1] = 1.0  # SZSE main board
    elif code.startswith("3"):
        meta[2] = 1.0  # ChiNext

    if window_features.shape[0] > 1:
        log_ret_col = 0  # log_return is first feature
        rets = window_features[:, log_ret_col]
        valid = rets[~np.isnan(rets) & (rets != 0)]
        if len(valid) > 0:
            meta[3] = 1.0 if np.any(np.abs(valid) > 0.09) else 0.0
            meta[4] = np.clip(np.std(valid) * 10, 0, 1)
            meta[5] = np.clip(np.sum(valid) * 2, -1, 1)

        vol_col = min(7, window_features.shape[1] - 1)  # volume_ma_ratio
        vol = window_features[:, vol_col]
        valid_vol = vol[~np.isnan(vol) & (vol != 0)]
        if len(valid_vol) > 0:
            meta[6] = np.clip(np.mean(valid_vol), 0, 3) / 3.0

        turn_col = min(10, window_features.shape[1] - 1)  # turnover_ratio
        turn = window_features[:, turn_col]
        valid_turn = turn[~np.isnan(turn) & (turn != 0)]
        if len(valid_turn) > 0:
            meta[7] = np.clip(np.mean(valid_turn), 0, 3) / 3.0

    return meta


class AStockFinetuneDataset(Dataset):
    """
    Dataset for A-share manipulation detection fine-tuning.
    Creates sliding windows over daily data with binary manipulation labels.
    """

    def __init__(
        self,
        cases: list[dict],
        astock_dir: str,
        feature_cfg: dict,
        seq_len: int = 60,
        stride: int = 5,
        min_bars: int = 30,
        overlap_ratio: float = 0.5,
        oversample_positive: int = 1,
        cot_embeddings: dict | None = None,
    ):
        self.seq_len = seq_len
        self.cot_embeddings = cot_embeddings or {}
        self.windows: list[dict] = []
        self.feature_names: list[str] = []

        for case_info in cases:
            loaded = load_case_data(case_info, astock_dir)
            if loaded is None:
                continue

            df, feat_names = compute_daily_features(
                loaded["df"],
                returns_horizons=feature_cfg.get("returns_horizons", [1, 5, 10, 20, 60]),
                volume_ma_window=feature_cfg.get("volume_ma_window", 20),
                rsi_period=feature_cfg.get("rsi_period", 14),
                bb_period=feature_cfg.get("bb_period", 20),
                atr_period=feature_cfg.get("atr_period", 14),
            )
            if not self.feature_names:
                self.feature_names = feat_names

            feat_matrix = df[feat_names].values.astype(np.float32)
            feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            dates = pd.to_datetime(df["date"]) if "date" in df.columns else df.index
            manip_start = loaded["manipulation_start"]
            manip_end = loaded["manipulation_end"]

            for start_idx in range(0, len(df) - min_bars, stride):
                end_idx = min(start_idx + seq_len, len(df))
                window_len = end_idx - start_idx

                if window_len < min_bars:
                    continue

                window_features = feat_matrix[start_idx:end_idx]
                window_features = normalize_window(window_features)

                if window_len < seq_len:
                    pad = np.zeros((seq_len - window_len, window_features.shape[1]), dtype=np.float32)
                    window_features = np.concatenate([window_features, pad], axis=0)
                    padding_mask = np.concatenate([
                        np.zeros(window_len, dtype=np.float32),
                        np.ones(seq_len - window_len, dtype=np.float32),
                    ])
                else:
                    padding_mask = np.zeros(seq_len, dtype=np.float32)

                window_dates = dates.iloc[start_idx:end_idx]
                manip_days = ((window_dates >= manip_start) & (window_dates <= manip_end)).sum()
                overlap = manip_days / window_len
                label = 1 if overlap >= overlap_ratio else 0

                cot_emb = self.cot_embeddings.get(loaded["doc_id"])

                try:
                    w_start_date = str(window_dates.iloc[0])[:10]
                except Exception:
                    w_start_date = ""

                meta = _build_metadata(
                    loaded["stock_code"], window_features, window_dates,
                )

                window_data = {
                    "features": window_features,
                    "padding_mask": padding_mask,
                    "label": label,
                    "doc_id": loaded["doc_id"],
                    "cot_embedding": cot_emb,
                    "stock_code": loaded["stock_code"],
                    "has_cot": cot_emb is not None,
                    "window_start_date": w_start_date,
                    "metadata": meta,
                }
                self.windows.append(window_data)

        n_pos = sum(1 for w in self.windows if w["label"] == 1)
        n_neg = len(self.windows) - n_pos

        if oversample_positive > 1:
            pos_windows = [w for w in self.windows if w["label"] == 1]
            neg_windows = [w for w in self.windows if w["label"] == 0]
            if n_pos < n_neg and pos_windows:
                target = max(n_neg, n_pos * oversample_positive)
                while len(pos_windows) < target:
                    pos_windows.extend(pos_windows[:target - len(pos_windows)])
                self.windows = neg_windows + pos_windows[:target]
            elif n_neg < n_pos and neg_windows:
                target = max(n_pos, n_neg * oversample_positive)
                while len(neg_windows) < target:
                    neg_windows.extend(neg_windows[:target - len(neg_windows)])
                self.windows = pos_windows + neg_windows[:target]

        n_pos = sum(1 for w in self.windows if w["label"] == 1)
        n_neg = len(self.windows) - n_pos

        from loguru import logger
        logger.info(
            f"AStockDataset: {len(self.windows)} windows "
            f"(pos={n_pos}, neg={n_neg}, ratio={n_pos/max(len(self.windows),1):.2%})"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        item = {
            "features": torch.from_numpy(w["features"]),
            "padding_mask": torch.from_numpy(w["padding_mask"]),
            "label": torch.tensor(w["label"], dtype=torch.long),
            "doc_id": w.get("doc_id", ""),
            "stock_code": w.get("stock_code", ""),
            "window_start_date": w.get("window_start_date", ""),
            "metadata": torch.from_numpy(w.get("metadata", np.zeros(8, dtype=np.float32))),
        }
        if w["cot_embedding"] is not None:
            item["cot_embedding"] = torch.from_numpy(w["cot_embedding"])
            item["has_cot"] = torch.tensor(1.0)
        else:
            item["cot_embedding"] = torch.zeros(384, dtype=torch.float32)
            item["has_cot"] = torch.tensor(0.0)
        return item

    def get_n_features(self) -> int:
        return len(self.feature_names) if self.feature_names else 25


def build_cv_splits(
    cases: list[dict],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[dict], list[dict]]]:
    """Create leave-cases-out cross-validation splits."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(cases))
    fold_size = len(cases) // n_folds

    splits = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(cases)
        val_indices = set(indices[start:end])
        train_cases = [cases[i] for i in range(len(cases)) if i not in val_indices]
        val_cases = [cases[i] for i in val_indices]
        splits.append((train_cases, val_cases))

    return splits
