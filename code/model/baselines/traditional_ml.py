"""
Traditional ML baselines for crypto anomaly detection.

Uses the same crypto data and anomaly labels as the MoE pre-training,
but applies Random Forest and XGBoost for comparison.

These baselines establish the performance floor that the MoE model
should outperform.
"""
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


def load_and_prepare_data(
    data_dir: str,
    symbols: list[str],
    seq_len: int = 256,
    stride: int = 64,
    anomaly_cfg: Optional[dict] = None,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load crypto data and create feature matrix for traditional ML.

    Instead of using raw time series windows, we aggregate each window
    into a fixed-size feature vector (statistical summaries).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.feature_engineering import compute_features, detect_anomalies, get_feature_names

    anomaly_cfg = anomaly_cfg or {}

    all_features = []
    all_labels = []

    for symbol in symbols:
        symbol_dir = Path(data_dir) / symbol
        if not symbol_dir.exists():
            continue

        parquet_files = sorted(symbol_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        df = compute_features(df)
        anomaly_flags = detect_anomalies(df, **anomaly_cfg)
        feature_cols = get_feature_names()
        feat_matrix = df[feature_cols].values.astype(np.float32)
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=3.0, neginf=-3.0)

        n = len(feat_matrix)
        lookahead = 10

        for start in range(0, n - seq_len - lookahead, stride):
            end = start + seq_len
            window = feat_matrix[start:end]

            agg = _aggregate_window(window)
            label = int(anomaly_flags[end:end + lookahead].any())

            all_features.append(agg)
            all_labels.append(label)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Time-based split
    split_idx = int(len(X) * (1 - val_ratio))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Data prepared: train={len(X_train)}, val={len(X_val)}, "
                f"pos_ratio_train={y_train.mean():.4f}, pos_ratio_val={y_val.mean():.4f}")

    return X_train, y_train, X_val, y_val


def _aggregate_window(window: np.ndarray) -> np.ndarray:
    """
    Aggregate a [seq_len, n_features] window into a flat feature vector.
    Computes: mean, std, min, max, last, trend (slope) for each feature.
    """
    agg = []
    for col_idx in range(window.shape[1]):
        col = window[:, col_idx]
        agg.extend([
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col),
            col[-1],                          # last value
            np.polyfit(range(len(col)), col, 1)[0] if len(col) > 1 else 0,  # trend
        ])
    return np.array(agg, dtype=np.float32)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train and evaluate Random Forest baseline."""
    logger.info("Training Random Forest...")
    t0 = time.time()

    # Handle class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale = neg_count / max(pos_count, 1)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        class_weight={0: 1.0, 1: min(scale, 10.0)},
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = rf.predict(X_val)
    y_prob = rf.predict_proba(X_val)[:, 1] if len(rf.classes_) > 1 else np.zeros(len(X_val))

    metrics = _compute_metrics(y_val, y_pred, y_prob)
    metrics["train_time_s"] = elapsed
    metrics["model"] = "RandomForest"

    logger.info(f"Random Forest: Acc={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
    return metrics


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train and evaluate XGBoost baseline."""
    logger.info("Training XGBoost...")
    t0 = time.time()

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale = neg_count / max(pos_count, 1)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=min(scale, 10.0),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0

    y_pred = xgb.predict(X_val)
    y_prob = xgb.predict_proba(X_val)[:, 1]

    metrics = _compute_metrics(y_val, y_pred, y_prob)
    metrics["train_time_s"] = elapsed
    metrics["model"] = "XGBoost"

    # Feature importance (top 20)
    importances = xgb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    metrics["top_features"] = [
        {"index": int(i), "importance": float(importances[i])} for i in top_idx
    ]

    logger.info(f"XGBoost: Acc={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
    return metrics


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def run_baselines(cfg: dict, output_dir: str = "outputs/baselines") -> dict:
    """Run all baseline experiments."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    anomaly_cfg = {
        "price_surge_threshold": data_cfg["anomaly"]["price_surge_threshold"],
        "volume_multiplier": data_cfg["anomaly"]["volume_multiplier"],
        "surge_window": data_cfg["anomaly"]["surge_window"],
        "min_gap": data_cfg["anomaly"]["min_gap_between"],
    }

    X_train, y_train, X_val, y_val = load_and_prepare_data(
        data_dir=data_cfg["raw_dir"],
        symbols=data_cfg["symbols"],
        seq_len=data_cfg["seq_len"],
        stride=data_cfg["stride"],
        anomaly_cfg=anomaly_cfg,
        val_ratio=data_cfg.get("val_ratio", 0.1),
    )

    results = {}
    results["rf"] = train_random_forest(X_train, y_train, X_val, y_val)
    results["xgb"] = train_xgboost(X_train, y_train, X_val, y_val)

    results["data_stats"] = {
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_features": int(X_train.shape[1]),
        "pos_ratio_train": float(y_train.mean()),
        "pos_ratio_val": float(y_val.mean()),
    }

    results_path = out_path / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Baseline results saved to {results_path}")

    return results
