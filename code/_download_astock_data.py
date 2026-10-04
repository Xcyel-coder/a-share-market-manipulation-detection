# -*- coding: utf-8 -*-
"""
Download daily OHLCV data for new manipulation cases using AkShare.
Creates extracted JSON metadata and daily.parquet for each case.
"""
import sys
import os
import json
import time
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

import akshare as ak

CSV_PATH = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\manipulation_cases_dataset.csv"
OUTPUT_BASE = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\data\raw\astock"
EXTRACTED_DIR = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\data\processed\extracted_cases"
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)

CONTEXT_BEFORE = 120  # trading days before manipulation start
CONTEXT_AFTER = 60    # trading days after manipulation end

df_cases = pd.read_csv(CSV_PATH)

# Filter: need valid dates, duration > 0, not CSRC_direct_only
valid_mask = (
    df_cases["start_date"].notna() &
    (df_cases["start_date"] != "") &
    df_cases["end_date"].notna() &
    (df_cases["end_date"] != "") &
    (df_cases["duration_days"] > 0) &
    (df_cases["source"] != "CSRC_direct_only")
)
new_cases = df_cases[valid_mask].reset_index(drop=True)
print(f"New cases to download: {len(new_cases)}")

# Generate doc_ids
results = {"success": [], "failed": [], "skipped": []}

for idx, row in new_cases.iterrows():
    stock_code = str(row["stock_code"])
    raw_code = stock_code.split(".")[0].zfill(6)
    doc_id = f"CASE-{idx+1:04d}"
    stock_name = str(row.get("stock_name", ""))

    start_date = pd.to_datetime(str(row["start_date"]))
    end_date = pd.to_datetime(str(row["end_date"]))

    case_dir = os.path.join(OUTPUT_BASE, doc_id)
    parquet_path = os.path.join(case_dir, "daily.parquet")
    if os.path.exists(parquet_path):
        print(f"  [{idx+1}/{len(new_cases)}] {doc_id} ({raw_code} {stock_name}) - already exists, skip")
        results["skipped"].append(doc_id)
        continue

    # Download range: extend context
    dl_start = (start_date - pd.Timedelta(days=CONTEXT_BEFORE * 2)).strftime("%Y%m%d")
    dl_end = (end_date + pd.Timedelta(days=CONTEXT_AFTER * 2)).strftime("%Y%m%d")

    print(f"  [{idx+1}/{len(new_cases)}] {doc_id}: {raw_code} ({stock_name}) "
          f"period={start_date.date()}~{end_date.date()}, download={dl_start}~{dl_end}")

    try:
        df_daily = ak.stock_zh_a_hist(
            symbol=raw_code,
            period="daily",
            start_date=dl_start,
            end_date=dl_end,
            adjust="qfq",
        )
        if df_daily is None or len(df_daily) == 0:
            print(f"    WARN: No data returned")
            results["failed"].append((doc_id, raw_code, "no data"))
            continue

        df_daily = df_daily.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turn",
            "涨跌幅": "pctChg",
        })

        market = "sh" if raw_code.startswith("6") else "sz"
        df_daily["code"] = f"{market}.{raw_code}"
        df_daily["stock_code"] = raw_code
        df_daily["market"] = market
        df_daily["date"] = pd.to_datetime(df_daily["date"])

        keep_cols = ["date", "code", "open", "high", "low", "close",
                     "volume", "amount", "turn", "pctChg", "stock_code", "market"]
        existing_cols = [c for c in keep_cols if c in df_daily.columns]
        df_daily = df_daily[existing_cols].sort_values("date").reset_index(drop=True)

        os.makedirs(case_dir, exist_ok=True)
        df_daily.to_parquet(parquet_path, index=False)

        # Create extracted JSON
        extracted = {
            "doc_id": doc_id,
            "case_number": str(row.get("penalty_id", "")),
            "defendant_name": "",
            "defendant_type": "",
            "stock_code": raw_code,
            "stock_name": stock_name,
            "manipulation_start_date": str(start_date.date()),
            "manipulation_end_date": str(end_date.date()),
            "manipulation_duration_days": int(row["duration_days"]),
            "manipulation_methods": ["操纵市场"],
            "manipulation_description": str(row.get("note", "")),
            "source": "third_party_excel",
            "original_stock_code": stock_code,
        }
        json_path = os.path.join(EXTRACTED_DIR, f"{doc_id}_extracted.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, ensure_ascii=False, indent=2)

        print(f"    OK: {len(df_daily)} bars, {df_daily['date'].min().date()}~{df_daily['date'].max().date()}")
        results["success"].append(doc_id)

    except Exception as e:
        print(f"    ERROR: {e}")
        results["failed"].append((doc_id, raw_code, str(e)))

    time.sleep(0.5)

print(f"\n{'='*60}")
print(f"Download complete:")
print(f"  Success: {len(results['success'])}")
print(f"  Failed: {len(results['failed'])}")
print(f"  Skipped: {len(results['skipped'])}")

if results["failed"]:
    print("\nFailed cases:")
    for item in results["failed"]:
        print(f"  {item}")

# Save results
results_path = os.path.join(OUTPUT_BASE, "_download_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "success": results["success"],
        "failed": [(d, c, e) for d, c, e in results["failed"]],
        "skipped": results["skipped"],
    }, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {results_path}")
