"""
Streamlined download: skip exploration, go straight to downloading key files.
Skips large crypto raw data (already local). Focus on weights, results, logs, code.
"""
import os
import paramiko, os, stat, sys
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 26391
USER = "root"
PASSWD = os.environ["AUTODL_SSH_PASSWORD"]
LOCAL = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\autodl_backup")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting...", flush=True)
ssh.connect(HOST, port=PORT, username=USER, password=PASSWD, timeout=30)
sftp = ssh.open_sftp()
print("OK\n", flush=True)

count = 0

def dl_file(remote, local_path):
    global count
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        st = sftp.stat(remote)
    except IOError:
        return
    if local_path.exists() and local_path.stat().st_size == st.st_size:
        return
    sz = st.st_size / 1024 / 1024
    print(f"  [{sz:7.2f} MB] {remote}", flush=True)
    sftp.get(remote, str(local_path))
    count += 1

def dl_dir(remote_dir, local_dir, skip_patterns=None):
    try:
        entries = sftp.listdir_attr(remote_dir)
    except IOError:
        return
    for e in entries:
        rp = f"{remote_dir}/{e.filename}"
        lp = local_dir / e.filename
        if skip_patterns and any(p in rp for p in skip_patterns):
            continue
        if stat.S_ISDIR(e.st_mode):
            dl_dir(rp, lp, skip_patterns)
        else:
            dl_file(rp, lp)

SKIP = ["/raw/crypto/", "/raw/raw/"]

print("=== 1/5: Model weights & results (outputs/) ===", flush=True)
dl_dir("/root/project/outputs", LOCAL / "outputs", SKIP)

print("\n=== 2/5: Data (cases & A-stock, skip crypto raw) ===", flush=True)
dl_dir("/root/project/data", LOCAL / "data", SKIP)

print("\n=== 3/5: Model code ===", flush=True)
dl_dir("/root/project/model", LOCAL / "model")

print("\n=== 4/5: Run scripts & logs ===", flush=True)
for e in sftp.listdir_attr("/root/project"):
    if not stat.S_ISDIR(e.st_mode):
        dl_file(f"/root/project/{e.filename}", LOCAL / "scripts" / e.filename)

print("\n=== 5/5: Pretrain weights (model/outputs) ===", flush=True)
dl_dir("/root/project/model/outputs", LOCAL / "model_outputs")

sftp.close()
ssh.close()
print(f"\nDONE: {count} new files downloaded to {LOCAL}", flush=True)
