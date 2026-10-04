# -*- coding: utf-8 -*-
"""
阶段一：差距分析
解析 manipulation_cases_dataset.csv 中的 penalty_id，
与已爬取的 CSRC 文档交叉比对，输出 A/B/C 三类案例清单。

A 类：有 penalty_id 且已爬取（可直接提取）
B 类：有 penalty_id 但未爬取（需定向爬取）
C 类：无 penalty_id（需按股票名搜索）
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文")
CSV_PATH = BASE / "manipulation_cases_dataset.csv"
CSRC_DIR = BASE / "code" / "data_pipeline" / "data" / "raw" / "csrc_penalties"
EXTRACTED_DIR = BASE / "code" / "data_pipeline" / "data" / "processed" / "extracted_cases"
OUTPUT_JSON = BASE / "code" / "_gap_analysis_result.json"


def penalty_id_to_csrc(penalty_id_raw) -> dict:
    """
    将 penalty_id（如 202244.0）转换为 CSRC 格式。
    Returns dict with year, number, csrc_id, and search_pattern.
    """
    if pd.isna(penalty_id_raw) or str(penalty_id_raw).strip() == "":
        return None

    s = str(penalty_id_raw).strip()
    # Already a CSRC ID (from CSRC_direct_only rows)
    if s.startswith("CSRC-"):
        return {"csrc_id": s, "year": None, "number": None, "search_pattern": s}

    # Numeric format like 202244.0 → year=2022, number=44
    s = s.replace(".0", "").replace(".", "")
    if not s.isdigit() or len(s) < 5:
        return None

    year = int(s[:4])
    number = int(s[4:])
    csrc_id = f"CSRC-{year}-{number:03d}"
    search_pattern = f"〔{year}〕{number}号"
    return {
        "csrc_id": csrc_id,
        "year": year,
        "number": number,
        "search_pattern": search_pattern,
    }


def load_crawled_docs() -> dict:
    """Load all crawled CSRC doc IDs from the index."""
    index_path = CSRC_DIR / "index.json"
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = {}
    for item in data.get("files", []):
        doc_id = item.get("doc_id", "")
        docs[doc_id] = {
            "title": item.get("title", ""),
            "case_type": item.get("case_type", ""),
            "url": item.get("url", ""),
            "keywords": item.get("keywords", []),
        }
    return docs


def load_extracted_cases() -> dict:
    """Load all LLM-extracted case data."""
    cases = {}
    if not EXTRACTED_DIR.exists():
        return cases
    for f in EXTRACTED_DIR.glob("*_extracted.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            doc_id = data.get("doc_id", f.stem.replace("_extracted", ""))
            cases[doc_id] = data
        except Exception:
            pass
    return cases


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    crawled = load_crawled_docs()
    extracted = load_extracted_cases()

    print("=" * 80)
    print("CSRC 标注验证 — 差距分析")
    print("=" * 80)
    print(f"CSV 总记录: {len(df)}")
    print(f"已爬取文档: {len(crawled)}")
    print(f"已提取案例: {len(extracted)}")

    # Skip CSRC_direct_only rows — these already have CSRC docs
    df_third = df[df["source"] != "CSRC_direct_only"].copy()
    print(f"第三方案例 (需验证): {len(df_third)}")

    cat_a, cat_b, cat_c = [], [], []

    for _, row in df_third.iterrows():
        rec = {
            "csv_id": int(row["id"]),
            "stock_code": row["stock_code"],
            "stock_name": row["stock_name"],
            "start_date": str(row["start_date"]) if pd.notna(row["start_date"]) else "",
            "end_date": str(row["end_date"]) if pd.notna(row["end_date"]) else "",
            "duration_days": int(row["duration_days"]),
            "penalty_id_raw": str(row["penalty_id"]) if pd.notna(row["penalty_id"]) else "",
        }

        parsed = penalty_id_to_csrc(row["penalty_id"])

        if parsed is None:
            rec["category"] = "C"
            rec["reason"] = "无 penalty_id，需按股票名搜索"
            cat_c.append(rec)
        else:
            csrc_id = parsed["csrc_id"]
            rec["csrc_id"] = csrc_id
            rec["search_pattern"] = parsed["search_pattern"]

            if csrc_id in crawled:
                rec["category"] = "A"
                rec["reason"] = "已爬取"
                rec["crawled_title"] = crawled[csrc_id]["title"]
                rec["has_extraction"] = csrc_id in extracted
                cat_a.append(rec)
            else:
                rec["category"] = "B"
                rec["reason"] = "有 penalty_id 但未爬取"
                cat_b.append(rec)

    # Print summary
    print()
    print("=" * 80)
    print(f"A 类 (有 penalty_id 且已爬取): {len(cat_a)}")
    print("=" * 80)
    for r in cat_a:
        ext = " [已提取]" if r.get("has_extraction") else " [未提取]"
        print(f"  {r['csv_id']:3d} {r['stock_code']} {r['stock_name']:8s} → {r['csrc_id']:16s}{ext}  {r['crawled_title'][:40]}")

    print()
    print("=" * 80)
    print(f"B 类 (有 penalty_id 但未爬取): {len(cat_b)}")
    print("=" * 80)
    for r in cat_b:
        print(f"  {r['csv_id']:3d} {r['stock_code']} {r['stock_name']:8s} → {r['csrc_id']:16s}  搜索: {r['search_pattern']}")

    print()
    print("=" * 80)
    print(f"C 类 (无 penalty_id，需搜索): {len(cat_c)}")
    print("=" * 80)
    for r in cat_c:
        dur = f"{r['duration_days']}天" if r["duration_days"] >= 0 else "N/A"
        print(f"  {r['csv_id']:3d} {r['stock_code']} {r['stock_name']:8s}  {r['start_date']} ~ {r['end_date']}  {dur}")

    # Suspicious long-duration cases
    suspicious = [r for r in (cat_b + cat_c) if r["duration_days"] > 1000]
    if suspicious:
        print()
        print("=" * 80)
        print(f"⚠ 可疑超长持续期案例 (>1000天): {len(suspicious)}")
        print("=" * 80)
        for r in suspicious:
            print(f"  {r['csv_id']:3d} {r['stock_code']} {r['stock_name']:8s}  {r['duration_days']}天  {r['start_date']} ~ {r['end_date']}")

    # Save result
    result = {
        "summary": {
            "total_third_party": len(df_third),
            "cat_a_count": len(cat_a),
            "cat_b_count": len(cat_b),
            "cat_c_count": len(cat_c),
            "suspicious_long_duration": len(suspicious) if suspicious else 0,
        },
        "cat_a": cat_a,
        "cat_b": cat_b,
        "cat_c": cat_c,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
