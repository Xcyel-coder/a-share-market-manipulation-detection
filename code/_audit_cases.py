import json
from pathlib import Path

cases_dir = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\data_pipeline\data\processed\extracted_cases")
astock_dir = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\model\data\raw\astock")

all_cases = []
for f in sorted(cases_dir.glob("*_extracted.json")):
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
    methods = c.get("manipulation_methods", [])
    doc_id = c.get("doc_id", "")
    has_astock = (astock_dir / doc_id / "daily.parquet").exists() if doc_id else False
    has_code = bool(c.get("stock_code"))
    has_dates = bool(c.get("manipulation_start_date") and c.get("manipulation_end_date"))
    methods_str = ", ".join(methods) if methods else "NONE"

    is_insider = any(k in methods_str for k in ["内幕交易", "利用未公开信息"])

    all_cases.append({
        "doc_id": doc_id,
        "stock_code": c.get("stock_code", ""),
        "stock_name": c.get("stock_name", ""),
        "methods": methods_str,
        "has_code": has_code,
        "has_dates": has_dates,
        "has_astock": has_astock,
        "trainable": has_code and has_dates and has_astock,
        "is_insider": is_insider,
    })

print(f"Total extracted JSON files: {len(all_cases)}")
trainable = [c for c in all_cases if c["trainable"]]
print(f"Trainable (code + dates + astock data): {len(trainable)}")

insider = [c for c in trainable if c["is_insider"]]
manip = [c for c in trainable if not c["is_insider"]]
print(f"  - Insider trading cases: {len(insider)}")
print(f"  - Manipulation cases: {len(manip)}")

print("\n=== TRAINABLE CASES (sorted by type) ===")
for c in sorted(trainable, key=lambda x: (x["is_insider"], x["doc_id"])):
    tag = "INSIDER" if c["is_insider"] else "MANIP  "
    print(f"  [{tag}] {c['doc_id']:25s} {c['stock_code']:8s} {c['stock_name']:12s} | {c['methods']}")

# Also check astock directories
print("\n=== ASTOCK DIRECTORIES ===")
astock_dirs = sorted(d.name for d in astock_dir.iterdir() if d.is_dir())
print(f"Total astock directories: {len(astock_dirs)}")
for d in astock_dirs:
    matched = any(c["doc_id"] == d for c in all_cases)
    print(f"  {d} {'(matched)' if matched else '(NO MATCH in extracted_cases)'}")
