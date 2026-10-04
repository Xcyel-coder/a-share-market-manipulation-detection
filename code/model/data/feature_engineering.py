"""
Feature engineering for financial time series.

Transforms raw OHLCV data into model-ready feature vectors.
All features are designed to be scale-invariant (ratios, log-returns, etc.)
so they transfer across markets (crypto → A-share).
"""
import numpy as np
import pandas as pd


FEATURE_COLUMNS: list[str] = []  # populated by compute_features()


def compute_features(
    df: pd.DataFrame,
    returns_horizons: list[int] = (1, 5, 15, 30, 60),
    volume_ma_window: int = 60,
    rsi_period: int = 14,
    bb_period: int = 20,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Compute all features from raw OHLCV DataFrame.

    Expects columns: timestamp, open, high, low, close, volume
    Optional columns: quote_volume, trades

    Returns DataFrame with original + feature columns.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    features = {}

    # --- Price-based features (scale-invariant) ---
    features["log_return"] = np.log(c / c.shift(1))
    features["high_low_pct"] = (h - l) / c
    features["close_open_pct"] = (c - o) / o.replace(0, np.nan)
    features["upper_shadow_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    features["lower_shadow_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c

    # --- Multi-horizon returns ---
    for horizon in returns_horizons:
        features[f"return_{horizon}m"] = c.pct_change(horizon)

    # --- Volume features ---
    vol_ma = v.rolling(volume_ma_window, min_periods=10).mean()
    vol_std = v.rolling(volume_ma_window, min_periods=10).std()
    features["volume_ma_ratio"] = v / vol_ma.replace(0, np.nan)
    features["volume_zscore"] = (v - vol_ma) / vol_std.replace(0, np.nan)
    features["volume_log_change"] = np.log(v / v.shift(1).replace(0, np.nan))

    if "quote_volume" in df.columns:
        vwap = df["quote_volume"] / v.replace(0, np.nan)
        features["vwap_deviation"] = (c - vwap) / c

    if "trades" in df.columns:
        trades_ma = df["trades"].rolling(volume_ma_window, min_periods=10).mean()
        features["trade_intensity"] = df["trades"] / trades_ma.replace(0, np.nan)

    # --- Volatility ---
    log_ret = features["log_return"]
    features["volatility_30m"] = log_ret.rolling(30, min_periods=5).std()
    features["volatility_60m"] = log_ret.rolling(60, min_periods=10).std()

    # --- RSI ---
    features["rsi"] = _compute_rsi(c, rsi_period)

    # --- Bollinger Band position ---
    bb_ma = c.rolling(bb_period, min_periods=5).mean()
    bb_std = c.rolling(bb_period, min_periods=5).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    features["bb_position"] = (c - bb_lower) / bb_range

    # --- MACD histogram (normalized) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features["macd_hist_norm"] = (macd_line - signal_line) / c

    # --- ATR ratio ---
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=5).mean()
    features["atr_ratio"] = atr / c

    # --- Time features (cyclical encoding) ---
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    else:
        dt = pd.to_datetime(df["timestamp"], unit="ms")
    hour_frac = dt.dt.hour + dt.dt.minute / 60.0
    features["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    dow = dt.dt.dayofweek.astype(float)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    for name, series in features.items():
        df[name] = series

    global FEATURE_COLUMNS
    FEATURE_COLUMNS = list(features.keys())

    return df


def detect_anomalies(
    df: pd.DataFrame,
    price_surge_threshold: float = 0.05,
    volume_multiplier: float = 3.0,
    surge_window: int = 5,
    min_gap: int = 30,
) -> np.ndarray:
    """
    Detect statistical anomalies (for contrastive learning labels).

    Returns binary array: 1 = anomaly timestep, 0 = normal.
    Uses lowered thresholds compared to Phase 1 P&D labeling.
    """
    n = len(df)
    labels = np.zeros(n, dtype=np.int64)

    c = df["close"].values
    v = df["volume"].values

    vol_ma = pd.Series(v).rolling(60, min_periods=10).mean().values

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


def normalize_window(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Instance normalization: zero mean, unit variance per feature within window.
    Shape: [seq_len, n_features] → [seq_len, n_features]
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + eps)


def get_feature_names() -> list[str]:
    """Return the list of computed feature column names."""
    return list(FEATURE_COLUMNS)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi / 100.0  # normalize to [0, 1]
