import json
from pathlib import Path

cases_dir = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\data_pipeline\data\processed\extracted_cases")

all_cases = []
for f in sorted(cases_dir.glob("*_extracted.json")):
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
    methods = c.get("manipulation_methods", [])
    doc_id = c.get("doc_id", "")
    methods_str = ", ".join(methods) if methods else "NONE"
    is_insider = any(k in methods_str for k in ["内幕交易", "利用未公开信息"])
    has_code = bool(c.get("stock_code"))
    has_dates = bool(c.get("manipulation_start_date") and c.get("manipulation_end_date"))

    all_cases.append({
        "doc_id": doc_id,
        "stock_code": c.get("stock_code") or "",
        "stock_name": c.get("stock_name") or "",
        "methods": methods_str,
        "is_insider": is_insider,
        "has_code": has_code,
        "has_dates": has_dates,
        "valid": has_code and has_dates,
    })

total = len(all_cases)
valid = [c for c in all_cases if c["valid"]]
insider_valid = [c for c in valid if c["is_insider"]]
manip_valid = [c for c in valid if not c["is_insider"]]

print(f"Total extracted files: {total}")
print(f"Valid (has code + dates): {len(valid)}")
print(f"  - Insider trading: {len(insider_valid)}")
print(f"  - Manipulation: {len(manip_valid)}")
print()

import csv, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=== INSIDER TRADING CASES ===")
for c in sorted(insider_valid, key=lambda x: x["doc_id"]):
    print(f"  {c['doc_id']:25s} {c['stock_code']:8s} {c['stock_name']} | {c['methods']}")

print()
print("=== MANIPULATION CASES ===")
for c in sorted(manip_valid, key=lambda x: x["doc_id"]):
    print(f"  {c['doc_id']:25s} {c['stock_code']:8s} {c['stock_name']} | {c['methods']}")

print()
print("=== INVALID CASES ===")
invalid = [c for c in all_cases if not c["valid"]]
for c in sorted(invalid, key=lambda x: x["doc_id"]):
    reason = []
    if not c["has_code"]: reason.append("no_code")
    if not c["has_dates"]: reason.append("no_dates")
    print(f"  {c['doc_id']:25s} {','.join(reason)} | {c['methods']}")
