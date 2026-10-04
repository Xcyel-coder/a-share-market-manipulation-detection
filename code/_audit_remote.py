"""Audit case data on AutoDL to check which cases are trainable and their types."""
import os
import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.westb.seetacloud.com", port=49397, username="root", password=os.environ["AUTODL_SSH_PASSWORD"], timeout=15)

# Count astock directories
_, out, _ = ssh.exec_command("ls -d /root/project/data/raw/astock/*/  | wc -l", timeout=10)
print(f"Astock directories: {out.read().decode().strip()}")

_, out, _ = ssh.exec_command("ls /root/project/data/raw/astock/", timeout=10)
print(f"Astock dirs: {out.read().decode().strip()}")

# Check which extracted cases have astock data
audit_script = r'''
import json, os
from pathlib import Path

cases_dir = Path("/root/project/data/processed/extracted_cases")
astock_dir = Path("/root/project/data/raw/astock")

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
        "trainable": has_code and has_dates and has_astock,
        "is_insider": is_insider,
        "has_astock": has_astock,
    })

trainable = [c for c in all_cases if c["trainable"]]
insider = [c for c in trainable if c["is_insider"]]
manip = [c for c in trainable if not c["is_insider"]]

print(f"Total extracted: {len(all_cases)}")
print(f"Trainable: {len(trainable)}")
print(f"  Insider: {len(insider)}")
print(f"  Manipulation: {len(manip)}")
print()
for c in sorted(trainable, key=lambda x: (x["is_insider"], x["doc_id"])):
    tag = "INSIDER" if c["is_insider"] else "MANIP  "
    print(f"[{tag}] {c['doc_id']:25s} {c['stock_code']:8s} {c['stock_name']:12s} | {c['methods']}")
'''

_, out, _ = ssh.exec_command(f'/root/miniconda3/bin/python -c "{audit_script}"', timeout=15)
print(out.read().decode())
_, err, _ = ssh.exec_command("", timeout=1)

ssh.close()
