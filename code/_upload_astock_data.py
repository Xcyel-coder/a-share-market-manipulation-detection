# -*- coding: utf-8 -*-
"""Upload new A-stock case data and extracted JSONs to the AutoDL server."""
import paramiko
import sys
import os
import json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

LOCAL_ASTOCK = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\data\raw\astock"
LOCAL_EXTRACTED = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\data\processed\extracted_cases"
REMOTE_ASTOCK = "/root/project/data/raw/astock"
REMOTE_EXTRACTED = "/root/project/data/processed/extracted_cases"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)
sftp = ssh.open_sftp()

# Upload CASE-* directories (parquet files)
case_dirs = sorted([d for d in os.listdir(LOCAL_ASTOCK) if d.startswith("CASE-")])
print(f"Uploading {len(case_dirs)} case directories...", flush=True)

uploaded, skipped = 0, 0
for case_dir in case_dirs:
    local_parquet = os.path.join(LOCAL_ASTOCK, case_dir, "daily.parquet")
    if not os.path.exists(local_parquet):
        continue

    remote_dir = f"{REMOTE_ASTOCK}/{case_dir}"
    remote_parquet = f"{remote_dir}/daily.parquet"

    try:
        sftp.stat(remote_parquet)
        skipped += 1
        continue
    except FileNotFoundError:
        pass

    try:
        sftp.mkdir(remote_dir)
    except:
        pass

    fsize = os.path.getsize(local_parquet)
    sftp.put(local_parquet, remote_parquet)
    uploaded += 1
    if uploaded % 10 == 0:
        print(f"  Uploaded {uploaded} cases...", flush=True)

print(f"  Parquet: uploaded={uploaded}, skipped={skipped}", flush=True)

# Upload extracted JSON files
json_files = sorted([f for f in os.listdir(LOCAL_EXTRACTED)
                     if f.startswith("CASE-") and f.endswith(".json")])
print(f"Uploading {len(json_files)} extracted JSON files...", flush=True)

json_uploaded = 0
for jf in json_files:
    local_json = os.path.join(LOCAL_EXTRACTED, jf)
    remote_json = f"{REMOTE_EXTRACTED}/{jf}"
    sftp.put(local_json, remote_json)
    json_uploaded += 1

print(f"  JSONs uploaded: {json_uploaded}", flush=True)

# Verify
stdin, stdout, stderr = ssh.exec_command(
    f"ls -d {REMOTE_ASTOCK}/CASE-* 2>/dev/null | wc -l", timeout=10)
count = stdout.read().decode().strip()
print(f"\nServer CASE-* directories: {count}", flush=True)

stdin, stdout, stderr = ssh.exec_command(
    f"ls {REMOTE_EXTRACTED}/CASE-*_extracted.json 2>/dev/null | wc -l", timeout=10)
count = stdout.read().decode().strip()
print(f"Server CASE-* extracted JSONs: {count}", flush=True)

stdin, stdout, stderr = ssh.exec_command(
    f"ls -d {REMOTE_ASTOCK}/*/ 2>/dev/null | wc -l", timeout=10)
total = stdout.read().decode().strip()
print(f"Server total astock dirs: {total}", flush=True)

sftp.close()
ssh.close()
print("Upload complete.", flush=True)
