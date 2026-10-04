# -*- coding: utf-8 -*-
"""
阶段四+五：正则提取 + 交叉校验 + 数据集重建

1. 从所有 CSRC 处罚决定书中用正则提取操纵日期、股票名、账户数等关键信息
2. 将提取结果与 CSV 数据集交叉校验
3. 分配质量层级 (Gold/Silver/Bronze)
4. 输出 manipulation_cases_verified.csv
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

P = lambda *a, **kw: print(*a, **kw, flush=True)

BASE = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文")
CSV_PATH = BASE / "manipulation_cases_dataset.csv"
CSRC_DIR = BASE / "code" / "data_pipeline" / "data" / "raw" / "csrc_penalties"
EXTRACTED_DIR = BASE / "code" / "data_pipeline" / "data" / "processed" / "extracted_cases"
OUTPUT_CSV = BASE / "manipulation_cases_verified.csv"
REPORT_JSON = BASE / "code" / "_verification_report.json"


# ─── Regex-based extraction from penalty text ─────────────────────────────────

def extract_dates_from_text(text: str) -> list[dict]:
    """Extract manipulation date ranges from penalty text using regex."""
    results = []

    # Pattern 1: YYYY年M月D日至YYYY年M月D日
    pat1 = r"(\d{4})年(\d{1,2})月(\d{1,2})日[至到](\d{4})年(\d{1,2})月(\d{1,2})日"
    for m in re.finditer(pat1, text):
        y1, m1, d1, y2, m2, d2 = m.groups()
        start = f"{y1}-{m1.zfill(2)}-{d1.zfill(2)}"
        end = f"{y2}-{m2.zfill(2)}-{d2.zfill(2)}"
        context = text[max(0, m.start()-30):m.end()+30]
        results.append({"start": start, "end": end, "context": context})

    # Pattern 2: YYYY年M月至YYYY年M月 (month-level only)
    pat2 = r"(\d{4})年(\d{1,2})月[至到](\d{4})年(\d{1,2})月"
    for m in re.finditer(pat2, text):
        y1, m1, y2, m2 = m.groups()
        start = f"{y1}-{m1.zfill(2)}-01"
        end = f"{y2}-{m2.zfill(2)}-28"
        context = text[max(0, m.start()-30):m.end()+30]
        if not any(r["start"] == start for r in results):
            results.append({"start": start, "end": end, "context": context, "approximate": True})

    return results


def extract_stock_names(text: str) -> list[str]:
    """Extract stock names from penalty text."""
    names = set()
    # "XX" pattern - stock names in Chinese quotes
    for m in re.finditer(r'[\u201c\u201d\u2018\u2019"]([\u4e00-\u9fa5A-Z*]{2,8})[\u201c\u201d\u2018\u2019"]', text):
        name = m.group(1)
        if any(kw in name for kw in ["证券", "账户", "公司", "银行", "委员会", "期货"]):
            continue
        names.add(name)

    # "操纵XX股票" pattern
    for m in re.finditer(r'操纵[\u201c\u201d"]*(.{2,8}?)[\u201c\u201d"]*(?:股票|股价|证券)', text):
        name = m.group(1).strip()
        if len(name) <= 8:
            names.add(name)

    return list(names)


def extract_account_count(text: str) -> int:
    """Extract number of accounts from penalty text."""
    for m in re.finditer(r"(\d+)个(?:证券)?账户", text):
        n = int(m.group(1))
        if 2 <= n <= 500:
            return n
    return 0


def extract_amounts(text: str) -> dict:
    """Extract financial amounts from penalty text."""
    result = {"illegal_profit": 0, "fine_amount": 0}

    # Illegal profit: 违法所得XX元
    for m in re.finditer(r"(?:违法所得|获利)[^\d]{0,10}([\d,]+(?:\.\d+)?)\s*元", text):
        val = float(m.group(1).replace(",", ""))
        if val > result["illegal_profit"]:
            result["illegal_profit"] = val

    # Fine: 罚款XX元
    for m in re.finditer(r"罚款[^\d]{0,10}([\d,]+(?:\.\d+)?)\s*元", text):
        val = float(m.group(1).replace(",", ""))
        if val > result["fine_amount"]:
            result["fine_amount"] = val

    return result


def extract_case_number(text: str) -> str:
    """Extract case/penalty number from text."""
    for pat in [r"〔(\d{4})〕(\d+)号", r"\[(\d{4})\](\d+)号"]:
        m = re.search(pat, text)
        if m:
            return f"〔{m.group(1)}〕{m.group(2)}号"
    return ""


def extract_manipulation_methods(text: str) -> list[str]:
    """Extract manipulation methods from penalty text."""
    methods = []
    kw_map = {
        "连续买卖": "连续交易", "连续交易": "连续交易",
        "集中资金优势": "集中资金优势", "持股优势": "持股优势",
        "对倒": "对倒交易", "对敲": "对倒交易",
        "拉抬": "拉抬股价", "打压": "打压股价",
        "虚假申报": "虚假申报", "撤单": "虚假申报",
        "尾市": "尾市操纵", "约定交易": "约定交易",
    }
    for kw, method in kw_map.items():
        if kw in text and method not in methods:
            methods.append(method)
    return methods


def extract_from_doc(doc_path: Path) -> dict | None:
    """Full extraction from a single CSRC penalty document."""
    try:
        data = json.loads(doc_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    content = data.get("content", "")
    title = data.get("title", "")
    text = title + "\n" + content

    if not content or len(content) < 300:
        return None

    dates = extract_dates_from_text(text)
    stocks = extract_stock_names(text)
    accounts = extract_account_count(text)
    amounts = extract_amounts(text)
    case_num = extract_case_number(text)
    methods = extract_manipulation_methods(text)

    # Find the "操纵期间" date range — prefer the one near manipulation keywords
    manipulation_date = None
    scored_dates = []
    for d in dates:
        ctx = d.get("context", "")
        score = 0
        if "操纵期间" in ctx or "操纵" in ctx:
            score += 10
        if "账户组" in ctx:
            score += 5
        if "连续买卖" in ctx or "集中资金" in ctx:
            score += 3
        # Prefer non-approximate dates
        if not d.get("approximate"):
            score += 2
        scored_dates.append((score, d))
    scored_dates.sort(key=lambda x: -x[0])
    if scored_dates:
        manipulation_date = scored_dates[0][1]

    return {
        "doc_id": data.get("doc_id", doc_path.stem),
        "title": title,
        "url": data.get("url", ""),
        "case_number": case_num,
        "publish_date": data.get("publish_date", ""),
        "stock_names": stocks,
        "manipulation_start": manipulation_date["start"] if manipulation_date else "",
        "manipulation_end": manipulation_date["end"] if manipulation_date else "",
        "all_date_ranges": dates,
        "account_count": accounts,
        "methods": methods,
        "illegal_profit": amounts["illegal_profit"],
        "fine_amount": amounts["fine_amount"],
        "case_type": data.get("case_type", ""),
        "keywords": data.get("keywords_matched", []),
    }


# ─── Cross-validation ─────────────────────────────────────────────────────────

def normalize_stock_name(name: str) -> str:
    """Normalize stock name for matching."""
    name = name.replace("ST ", "").replace("*ST ", "").replace("ST", "")
    return name.strip()


def match_csrc_to_csv(csrc_case: dict, csv_row: pd.Series) -> dict | None:
    """Try to match a CSRC case to a CSV row by stock name."""
    csv_name = normalize_stock_name(str(csv_row["stock_name"]))
    csrc_names = [normalize_stock_name(n) for n in csrc_case["stock_names"]]

    # Also check stock names in title
    title = csrc_case.get("title", "")
    for n in csrc_names:
        if n in csv_name or csv_name in n:
            return csrc_case

    if csv_name in title:
        return csrc_case

    return None


def compute_date_deviation(csv_start: str, csv_end: str,
                           csrc_start: str, csrc_end: str) -> dict:
    """Compute date deviation between CSV and CSRC dates."""
    result = {
        "start_dev_days": None, "end_dev_days": None,
        "max_dev_days": None, "match_level": "no_csrc_data",
    }

    try:
        if csv_start and csrc_start:
            d1 = datetime.strptime(csv_start, "%Y-%m-%d")
            d2 = datetime.strptime(csrc_start, "%Y-%m-%d")
            result["start_dev_days"] = abs((d2 - d1).days)

        if csv_end and csrc_end:
            d1 = datetime.strptime(csv_end, "%Y-%m-%d")
            d2 = datetime.strptime(csrc_end, "%Y-%m-%d")
            result["end_dev_days"] = abs((d2 - d1).days)

        devs = [d for d in [result["start_dev_days"], result["end_dev_days"]] if d is not None]
        if devs:
            result["max_dev_days"] = max(devs)
            if result["max_dev_days"] <= 7:
                result["match_level"] = "exact"
            elif result["max_dev_days"] <= 30:
                result["match_level"] = "minor_deviation"
            else:
                result["match_level"] = "major_deviation"
    except Exception:
        pass

    return result


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    P("=" * 80)
    P("CSRC 标注验证与数据集重建")
    P("=" * 80)

    # Step 1: Extract from all CSRC docs
    P("\n--- 步骤一：从所有 CSRC 文档中提取关键信息 ---")
    all_extractions = {}
    doc_count = 0
    for f in CSRC_DIR.rglob("*.json"):
        if f.name == "index.json":
            continue
        ext = extract_from_doc(f)
        if ext and ext["case_type"] == "操纵证券市场":
            all_extractions[ext["doc_id"]] = ext
            doc_count += 1

    # Also load existing LLM extractions
    llm_extractions = {}
    if EXTRACTED_DIR.exists():
        for f in EXTRACTED_DIR.glob("*_extracted.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                doc_id = data.get("doc_id", f.stem.replace("_extracted", ""))
                llm_extractions[doc_id] = data
            except Exception:
                pass

    P(f"操纵类文档提取: {doc_count}")
    P(f"已有LLM提取: {len(llm_extractions)}")

    # Step 2: Build multiple lookup indexes
    P("\n--- 步骤二：建立多维度案例映射 ---")
    name_to_csrc = {}
    docid_to_csrc = {}
    for doc_id, ext in all_extractions.items():
        docid_to_csrc[doc_id] = ext
        for name in ext["stock_names"]:
            clean = normalize_stock_name(name)
            if clean and len(clean) >= 2:
                name_to_csrc.setdefault(clean, []).append(ext)

    P(f"股票名索引: {len(name_to_csrc)} 个名称, doc_id索引: {len(docid_to_csrc)}")

    # Step 3: Load CSV and match
    P("\n--- 步骤三：交叉校验 CSV 与 CSRC 提取 ---")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    verified_rows = []
    stats = {
        "gold": 0, "silver": 0, "bronze": 0,
        "exact_match": 0, "minor_dev": 0, "major_dev": 0, "no_csrc": 0,
    }

    for _, row in df.iterrows():
        rec = {
            "id": int(row["id"]),
            "stock_code": row["stock_code"],
            "stock_name": row["stock_name"],
            "original_start_date": str(row["start_date"]) if pd.notna(row["start_date"]) else "",
            "original_end_date": str(row["end_date"]) if pd.notna(row["end_date"]) else "",
            "duration_days": int(row["duration_days"]),
            "penalty_date": str(row["penalty_date"]) if pd.notna(row["penalty_date"]) else "",
            "penalty_id": str(row["penalty_id"]) if pd.notna(row["penalty_id"]) else "",
            "source": row["source"],
            "note": str(row["note"]) if pd.notna(row["note"]) else "",
        }

        # Try to find matching CSRC case
        matched_csrc = None
        matched_llm = None
        csv_name = normalize_stock_name(str(row["stock_name"]))

        # Direct match by CSRC_direct_only doc_id
        if row["source"] == "CSRC_direct_only" and pd.notna(row["penalty_id"]):
            doc_id = str(row["penalty_id"]).strip()
            if doc_id in all_extractions:
                matched_csrc = all_extractions[doc_id]
            if doc_id in llm_extractions:
                matched_llm = llm_extractions[doc_id]

        # For third-party cases with penalty_id, try doc_id lookup
        if not matched_csrc and pd.notna(row["penalty_id"]) and str(row["penalty_id"]).strip():
            pid = str(row["penalty_id"]).strip()
            # Convert 202244.0 → CSRC-2022-044
            pid_clean = pid.replace(".0", "").replace(".", "")
            if pid_clean.isdigit() and len(pid_clean) >= 5:
                year = pid_clean[:4]
                num = int(pid_clean[4:])
                csrc_id = f"CSRC-{year}-{num:03d}"
                if csrc_id in docid_to_csrc:
                    matched_csrc = docid_to_csrc[csrc_id]
                elif csrc_id in llm_extractions:
                    matched_llm = llm_extractions[csrc_id]

        # Match by stock name
        if not matched_csrc and csv_name in name_to_csrc:
            candidates = name_to_csrc[csv_name]
            if len(candidates) == 1:
                matched_csrc = candidates[0]
            else:
                best = None
                best_dev = float("inf")
                for c in candidates:
                    if c["manipulation_start"] and rec["original_start_date"]:
                        try:
                            d1 = datetime.strptime(rec["original_start_date"], "%Y-%m-%d")
                            d2 = datetime.strptime(c["manipulation_start"], "%Y-%m-%d")
                            dev = abs((d2 - d1).days)
                            if dev < best_dev:
                                best_dev = dev
                                best = c
                        except Exception:
                            pass
                if best:
                    matched_csrc = best

        # Compute verification
        if matched_csrc:
            csrc_start = matched_csrc["manipulation_start"]
            csrc_end = matched_csrc["manipulation_end"]

            # Also check LLM extraction for this doc
            doc_id = matched_csrc["doc_id"]
            if doc_id in llm_extractions:
                llm = llm_extractions[doc_id]
                if not csrc_start and llm.get("manipulation_start_date"):
                    csrc_start = llm["manipulation_start_date"]
                if not csrc_end and llm.get("manipulation_end_date"):
                    csrc_end = llm["manipulation_end_date"]

            rec["csrc_doc_id"] = doc_id
            rec["csrc_start_date"] = csrc_start
            rec["csrc_end_date"] = csrc_end
            rec["csrc_methods"] = ", ".join(matched_csrc.get("methods", []))
            rec["csrc_account_count"] = matched_csrc.get("account_count", 0)
            rec["csrc_illegal_profit"] = matched_csrc.get("illegal_profit", 0)
            rec["csrc_case_number"] = matched_csrc.get("case_number", "")
            rec["csrc_title"] = matched_csrc.get("title", "")[:60]

            deviation = compute_date_deviation(
                rec["original_start_date"], rec["original_end_date"],
                csrc_start, csrc_end)
            rec["start_deviation_days"] = deviation["start_dev_days"]
            rec["end_deviation_days"] = deviation["end_dev_days"]
            rec["max_deviation_days"] = deviation["max_dev_days"]

            # Use CSRC dates as verified dates
            rec["verified_start_date"] = csrc_start or rec["original_start_date"]
            rec["verified_end_date"] = csrc_end or rec["original_end_date"]

            # Assign quality tier
            level = deviation["match_level"]
            if level == "exact":
                rec["quality_tier"] = "Gold"
                stats["gold"] += 1
                stats["exact_match"] += 1
            elif level == "minor_deviation":
                rec["quality_tier"] = "Gold"
                stats["gold"] += 1
                stats["minor_dev"] += 1
            elif level == "major_deviation":
                rec["quality_tier"] = "Silver"
                stats["silver"] += 1
                stats["major_dev"] += 1
            else:
                # CSRC found but dates couldn't be compared
                if row["source"] == "CSRC_direct_only":
                    rec["quality_tier"] = "Gold"
                    stats["gold"] += 1
                else:
                    rec["quality_tier"] = "Silver"
                    stats["silver"] += 1
        else:
            # No CSRC match
            rec["csrc_doc_id"] = ""
            rec["csrc_start_date"] = ""
            rec["csrc_end_date"] = ""
            rec["csrc_methods"] = ""
            rec["csrc_account_count"] = 0
            rec["csrc_illegal_profit"] = 0
            rec["csrc_case_number"] = ""
            rec["csrc_title"] = ""
            rec["start_deviation_days"] = None
            rec["end_deviation_days"] = None
            rec["max_deviation_days"] = None
            rec["verified_start_date"] = rec["original_start_date"]
            rec["verified_end_date"] = rec["original_end_date"]

            if row["source"] == "CSRC_direct_only":
                rec["quality_tier"] = "Gold"
                stats["gold"] += 1
            elif pd.notna(row["penalty_id"]) and str(row["penalty_id"]).strip():
                rec["quality_tier"] = "Silver"
                stats["silver"] += 1
            else:
                rec["quality_tier"] = "Bronze"
                stats["bronze"] += 1
            stats["no_csrc"] += 1

        verified_rows.append(rec)

    # Step 4: Output
    P("\n--- 步骤四：输出验证后数据集 ---")

    out_df = pd.DataFrame(verified_rows)
    col_order = [
        "id", "stock_code", "stock_name",
        "verified_start_date", "verified_end_date", "duration_days",
        "original_start_date", "original_end_date",
        "quality_tier", "source",
        "csrc_doc_id", "csrc_case_number", "csrc_start_date", "csrc_end_date",
        "csrc_methods", "csrc_account_count", "csrc_illegal_profit",
        "start_deviation_days", "end_deviation_days", "max_deviation_days",
        "penalty_date", "penalty_id", "csrc_title", "note",
    ]
    existing_cols = [c for c in col_order if c in out_df.columns]
    out_df = out_df[existing_cols]
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    P(f"验证后数据集: {OUTPUT_CSV}")
    P(f"总记录数: {len(out_df)}")

    # Summary
    P()
    P("=" * 80)
    P("验证总结")
    P("=" * 80)
    P(f"Gold  (CSRC验证，日期偏差<=30天): {stats['gold']}")
    P(f"Silver (有CSRC参考/有penalty_id): {stats['silver']}")
    P(f"Bronze (无官方佐证): {stats['bronze']}")
    P()
    P(f"日期完全匹配 (<=7天): {stats['exact_match']}")
    P(f"轻微偏差 (8-30天): {stats['minor_dev']}")
    P(f"重大偏差 (>30天): {stats['major_dev']}")
    P(f"无CSRC数据: {stats['no_csrc']}")

    # Per-tier breakdown
    gold_df = out_df[out_df["quality_tier"] == "Gold"]
    silver_df = out_df[out_df["quality_tier"] == "Silver"]
    bronze_df = out_df[out_df["quality_tier"] == "Bronze"]

    P()
    P(f"Gold 案例 ({len(gold_df)}):")
    for _, r in gold_df.iterrows():
        dev = r.get("max_deviation_days")
        dev_str = f"偏差{int(dev)}天" if pd.notna(dev) else "CSRC直采"
        P(f"  {r['id']:3d} {r['stock_code']} {r['stock_name']:8s} {r['verified_start_date']}~{r['verified_end_date']}  {dev_str}")

    P()
    P(f"Silver 案例 ({len(silver_df)}):")
    for _, r in silver_df.iterrows():
        dev = r.get("max_deviation_days")
        dev_str = f"偏差{int(dev)}天" if pd.notna(dev) else "未匹配"
        P(f"  {r['id']:3d} {r['stock_code']} {r['stock_name']:8s} {dev_str}")

    P()
    P(f"Bronze 案例 ({len(bronze_df)}):")
    for _, r in bronze_df.iterrows():
        P(f"  {r['id']:3d} {r['stock_code']} {r['stock_name']:8s} {r['original_start_date']}~{r['original_end_date']}")

    # Suspicious cases (>1000 days, should be reviewed)
    suspicious = out_df[out_df["duration_days"] > 1000]
    if len(suspicious) > 0:
        P()
        P(f"⚠ 可疑超长案例 ({len(suspicious)} 个，建议审查):")
        for _, r in suspicious.iterrows():
            P(f"  {r['id']:3d} {r['stock_code']} {r['stock_name']:8s} {r['duration_days']}天 [{r['quality_tier']}]")

    # Save verification report
    report = {
        "summary": stats,
        "total_records": len(out_df),
        "quality_distribution": {
            "Gold": len(gold_df),
            "Silver": len(silver_df),
            "Bronze": len(bronze_df),
        },
        "csrc_extractions_total": doc_count,
        "llm_extractions_total": len(llm_extractions),
        "suspicious_long_duration": len(suspicious),
        "verification_time": datetime.now().isoformat(),
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    P(f"\n验证报告: {REPORT_JSON}")


if __name__ == "__main__":
    main()
