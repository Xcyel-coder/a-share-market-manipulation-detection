# -*- coding: utf-8 -*-
"""
Download daily OHLCV data for new manipulation cases.
Uses BaoStock (more stable, supports batch queries).
"""
import sys
import os
import json
import time
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

import baostock as bs

CSV_PATH = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\manipulation_cases_dataset.csv"
OUTPUT_BASE = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\data\raw\astock"
EXTRACTED_DIR = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\data\processed\extracted_cases"
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)

CONTEXT_BEFORE = 120
CONTEXT_AFTER = 60

def p(msg):
    print(msg, flush=True)

p("Logging into BaoStock...")
lg = bs.login()
p(f"Login: {lg.error_code} - {lg.error_msg}")

df_cases = pd.read_csv(CSV_PATH)
valid_mask = (
    df_cases["start_date"].notna() &
    (df_cases["start_date"] != "") &
    df_cases["end_date"].notna() &
    (df_cases["end_date"] != "") &
    (df_cases["duration_days"] > 0) &
    (df_cases["source"] != "CSRC_direct_only")
)
new_cases = df_cases[valid_mask].reset_index(drop=True)
p(f"Cases to download: {len(new_cases)}")

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
        p(f"  [{idx+1}/{len(new_cases)}] {doc_id} ({raw_code} {stock_name}) - exists, skip")
        results["skipped"].append(doc_id)
        continue

    # BaoStock format: sh.600000 or sz.000001
    if raw_code.startswith("6"):
        bs_code = f"sh.{raw_code}"
        market = "sh"
    else:
        bs_code = f"sz.{raw_code}"
        market = "sz"

    dl_start = (start_date - pd.Timedelta(days=CONTEXT_BEFORE * 2)).strftime("%Y-%m-%d")
    dl_end = (end_date + pd.Timedelta(days=CONTEXT_AFTER * 2)).strftime("%Y-%m-%d")

    p(f"  [{idx+1}/{len(new_cases)}] {doc_id}: {bs_code} ({stock_name}) "
      f"{start_date.date()}~{end_date.date()}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg",
                start_date=dl_start,
                end_date=dl_end,
                frequency="d",
                adjustflag="2",  # 前复权
            )

            if rs.error_code != "0":
                p(f"    API error: {rs.error_code} - {rs.error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                results["failed"].append((doc_id, raw_code, rs.error_msg))
                break

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                p(f"    WARN: No data returned")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                results["failed"].append((doc_id, raw_code, "no data"))
                break

            df_daily = pd.DataFrame(data_list, columns=rs.fields)

            numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
            for c in numeric_cols:
                if c in df_daily.columns:
                    df_daily[c] = pd.to_numeric(df_daily[c], errors="coerce")

            df_daily["date"] = pd.to_datetime(df_daily["date"])
            df_daily["stock_code"] = raw_code
            df_daily["market"] = market
            df_daily = df_daily.sort_values("date").reset_index(drop=True)

            # Remove rows with 0 close price
            df_daily = df_daily[df_daily["close"] > 0].reset_index(drop=True)

            os.makedirs(case_dir, exist_ok=True)
            df_daily.to_parquet(parquet_path, index=False)

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

            p(f"    OK: {len(df_daily)} bars, {df_daily['date'].min().date()}~{df_daily['date'].max().date()}")
            results["success"].append(doc_id)
            break

        except Exception as e:
            p(f"    Attempt {attempt+1} error: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                results["failed"].append((doc_id, raw_code, str(e)))

    time.sleep(0.5)

bs.logout()

p(f"\n{'='*60}")
p(f"Download complete:")
p(f"  Success: {len(results['success'])}")
p(f"  Failed: {len(results['failed'])}")
p(f"  Skipped: {len(results['skipped'])}")

if results["failed"]:
    p("\nFailed cases:")
    for item in results["failed"]:
        p(f"  {item}")

results_json = {
    "success": results["success"],
    "failed": [(d, c, e) for d, c, e in results["failed"]],
    "skipped": results["skipped"],
}
with open(os.path.join(OUTPUT_BASE, "_download_results.json"), "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
p("Done.")
