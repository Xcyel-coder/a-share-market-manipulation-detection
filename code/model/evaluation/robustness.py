"""
Robustness tests: time window, market cap, industry, manipulation type.
"""
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger


INDUSTRY_MAP = {
    "000": "其他", "001": "农林牧渔", "002": "采矿",
    "300": "创业板", "600": "主板", "601": "主板", "603": "主板",
}

SH_INDUSTRIES = {
    "600": "主板沪", "601": "主板沪", "603": "主板沪", "605": "主板沪",
}


def get_stock_info(stock_code: str, astock_dir: str) -> dict:
    """Extract industry proxy and market cap proxy from stock data."""
    info = {"industry_proxy": "unknown", "market_cap_proxy": "unknown"}

    if stock_code.startswith("6"):
        info["exchange"] = "上交所"
        info["board"] = "主板" if stock_code.startswith(("600", "601", "603")) else "科创板"
    elif stock_code.startswith("0"):
        info["exchange"] = "深交所"
        info["board"] = "主板"
    elif stock_code.startswith("3"):
        info["exchange"] = "深交所"
        info["board"] = "创业板"

    parquet_path = Path(astock_dir) / stock_code / "daily.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            if "close" in df.columns and "volume" in df.columns and len(df) > 0:
                avg_turnover = (df["close"] * df["volume"]).tail(60).mean()
                if avg_turnover > 1e9:
                    info["market_cap_proxy"] = "大市值"
                elif avg_turnover > 2e8:
                    info["market_cap_proxy"] = "中市值"
                else:
                    info["market_cap_proxy"] = "小市值"
        except Exception:
            pass

    return info


def split_by_time(cases, cutoff_year=2019):
    """Split cases into early (before cutoff) and late (after cutoff) periods."""
    early, late = [], []
    for c in cases:
        start = c.get("manipulation_start_date", "")
        try:
            year = int(start[:4])
            if year < cutoff_year:
                early.append(c)
            else:
                late.append(c)
        except (ValueError, TypeError):
            late.append(c)
    return {"2014-2018": early, "2019-2024": late}


def split_by_market_cap(cases, astock_dir):
    """Split cases by market cap proxy (based on average daily turnover)."""
    groups = defaultdict(list)
    for c in cases:
        info = get_stock_info(c["stock_code"], astock_dir)
        cap = info.get("market_cap_proxy", "unknown")
        groups[cap].append(c)
    return dict(groups)


def split_by_manipulation_type(cases):
    """Split cases by primary manipulation method."""
    type_map = {
        "连续交易": "连续交易",
        "对倒": "对倒交易",
        "对敲": "对倒交易",
        "虚假申报": "虚假申报",
        "撤单": "虚假申报",
        "拉抬": "拉抬操纵",
        "尾市": "尾市操纵",
        "抢帽子": "抢帽子交易",
        "利用信息优势": "信息优势操纵",
    }

    groups = defaultdict(list)
    for c in cases:
        methods = c.get("manipulation_methods", [])
        assigned = False
        for method_str in methods:
            for keyword, category in type_map.items():
                if keyword in method_str:
                    groups[category].append(c)
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            groups["其他"].append(c)

    return dict(groups)


def split_by_board(cases):
    """Split cases by stock exchange board type."""
    groups = defaultdict(list)
    for c in cases:
        code = c.get("stock_code", "")
        if code.startswith("6"):
            groups["沪市主板"].append(c)
        elif code.startswith("0"):
            groups["深市主板"].append(c)
        elif code.startswith("3"):
            groups["创业板"].append(c)
        else:
            groups["其他"].append(c)
    return dict(groups)


def summarize_case_metadata(cases, astock_dir):
    """Generate metadata summary for all cases."""
    time_groups = split_by_time(cases)
    cap_groups = split_by_market_cap(cases, astock_dir)
    type_groups = split_by_manipulation_type(cases)
    board_groups = split_by_board(cases)

    summary = {
        "total_cases": len(cases),
        "time_distribution": {k: len(v) for k, v in time_groups.items()},
        "market_cap_distribution": {k: len(v) for k, v in cap_groups.items()},
        "manipulation_type_distribution": {k: len(v) for k, v in type_groups.items()},
        "board_distribution": {k: len(v) for k, v in board_groups.items()},
    }

    durations = [c.get("manipulation_duration_days", 0) for c in cases]
    summary["duration_stats"] = {
        "mean": float(np.mean(durations)),
        "median": float(np.median(durations)),
        "min": int(np.min(durations)),
        "max": int(np.max(durations)),
    }

    logger.info(f"Case metadata summary:")
    for key, dist in summary.items():
        if isinstance(dist, dict) and key != "duration_stats":
            logger.info(f"  {key}: {dict(dist)}")

    return summary
