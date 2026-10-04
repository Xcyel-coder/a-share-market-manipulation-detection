"""
Simple vectorized backtesting framework.
Strategy: Model warns manipulation -> T+1 sell at open -> compute avoided losses.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def run_backtest(cases, predictions_by_case, astock_dir, hold_days=5):
    """
    Vectorized backtest for manipulation avoidance.
    
    predictions_by_case: dict[doc_id -> list of (date_str, pred, prob)]
    For each case where model predicts manipulation:
      - Compute T+1 to T+hold_days return (the loss avoided by selling)
    """
    results = []

    for case in cases:
        doc_id = case.get("doc_id", "")
        stock_code = case.get("stock_code", "")
        manip_start = case.get("manipulation_start_date", "")
        manip_end = case.get("manipulation_end_date", "")

        if doc_id not in predictions_by_case:
            continue

        parquet_path = Path(astock_dir) / doc_id / "daily.parquet"
        if not parquet_path.exists():
            continue

        try:
            df = pd.read_parquet(parquet_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            continue

        preds = predictions_by_case[doc_id]
        first_warning_date = None

        for date_str, pred, prob in sorted(preds, key=lambda x: x[0]):
            if pred == 1 and prob > 0.5:
                first_warning_date = date_str
                break

        if first_warning_date is None:
            results.append({
                "doc_id": doc_id, "stock_code": stock_code,
                "warned": False, "avoided_loss_pct": 0.0,
            })
            continue

        warn_idx = df[df["date"] >= pd.to_datetime(first_warning_date)].index
        if len(warn_idx) == 0:
            continue

        sell_idx = warn_idx[0] + 1
        if sell_idx >= len(df):
            continue

        sell_price = df.loc[sell_idx, "open"] if "open" in df.columns else df.loc[sell_idx, "close"]

        end_idx = min(sell_idx + hold_days, len(df) - 1)
        manip_end_idx = df[df["date"] >= pd.to_datetime(manip_end)].index
        if len(manip_end_idx) > 0:
            end_idx = min(manip_end_idx[0] + 1, len(df) - 1)

        end_price = df.loc[end_idx, "close"]
        loss_if_held = (end_price - sell_price) / (sell_price + 1e-9)

        manip_peak_idx = df[(df["date"] >= pd.to_datetime(manip_start)) &
                            (df["date"] <= pd.to_datetime(manip_end))].index
        if len(manip_peak_idx) > 0:
            peak_price = df.loc[manip_peak_idx, "close"].max()
            post_manip_idx = manip_peak_idx[-1] + 5 if manip_peak_idx[-1] + 5 < len(df) else len(df) - 1
            post_price = df.loc[post_manip_idx, "close"]
            crash_pct = (post_price - peak_price) / (peak_price + 1e-9)
        else:
            crash_pct = 0.0

        lead_days = (pd.to_datetime(manip_end) - pd.to_datetime(first_warning_date)).days

        results.append({
            "doc_id": doc_id,
            "stock_code": stock_code,
            "stock_name": case.get("stock_name", ""),
            "warned": True,
            "warning_date": first_warning_date,
            "manip_start": manip_start,
            "manip_end": manip_end,
            "sell_price": float(sell_price),
            "end_price": float(end_price),
            "loss_if_held_pct": float(loss_if_held * 100),
            "crash_pct": float(crash_pct * 100),
            "lead_days": lead_days,
        })

    warned = [r for r in results if r.get("warned")]
    not_warned = [r for r in results if not r.get("warned")]

    avoided_losses = [r["loss_if_held_pct"] for r in warned if r["loss_if_held_pct"] < 0]
    crash_pcts = [r["crash_pct"] for r in warned]

    summary = {
        "total_cases": len(results),
        "cases_warned": len(warned),
        "cases_not_warned": len(not_warned),
        "warning_rate": len(warned) / max(len(results), 1),
        "avg_loss_if_held_pct": float(np.mean([r["loss_if_held_pct"] for r in warned])) if warned else 0,
        "avg_avoided_loss_pct": float(np.mean(avoided_losses)) if avoided_losses else 0,
        "avg_crash_after_manip_pct": float(np.mean(crash_pcts)) if crash_pcts else 0,
        "avg_lead_days": float(np.mean([r["lead_days"] for r in warned])) if warned else 0,
        "details": results,
    }

    logger.info(f"Backtest: {summary['cases_warned']}/{summary['total_cases']} cases warned, "
                f"avg avoided loss: {summary['avg_avoided_loss_pct']:.2f}%")

    return summary
