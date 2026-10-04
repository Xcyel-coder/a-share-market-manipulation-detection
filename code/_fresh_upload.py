"""Fresh tar.gz creation + upload + extraction."""
import sys
import os
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

LOCAL_CRYPTO = Path(__file__).parent / "data_pipeline" / "data" / "raw" / "crypto"
tar_path = os.path.join(os.environ["TEMP"], "crypto_data.tar.gz")

# Step 1: Create tar.gz
print("Step 1: Creating tar.gz...", flush=True)
parquet_files = list(LOCAL_CRYPTO.rglob("*.parquet"))
print(f"  {len(parquet_files)} files found", flush=True)

with tarfile.open(tar_path, "w:gz") as tar:
    for i, pf in enumerate(sorted(parquet_files)):
        arcname = f"crypto/{pf.relative_to(LOCAL_CRYPTO).as_posix()}"
        tar.add(str(pf), arcname=arcname)
        if (i + 1) % 5 == 0:
            print(f"  Compressed {i + 1}/{len(parquet_files)}", flush=True)

size_mb = os.path.getsize(tar_path) / 1024 / 1024
print(f"  Done: {size_mb:.1f} MB", flush=True)

# Step 2: Upload
print("\nStep 2: Uploading...", flush=True)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
sftp = ssh.open_sftp()

last_pct = [0]
def progress(transferred, total):
    pct = int(transferred / total * 100)
    if pct >= last_pct[0] + 10:
        last_pct[0] = pct
        mb_done = transferred / 1024 / 1024
        mb_total = total / 1024 / 1024
        print(f"  {pct}% ({mb_done:.0f}/{mb_total:.0f} MB)", flush=True)

sftp.put(tar_path, "/root/project/crypto_data.tar.gz", callback=progress)
print("  Upload complete!", flush=True)
sftp.close()

# Step 3: Extract on remote
print("\nStep 3: Extracting...", flush=True)
stdin, stdout, stderr = ssh.exec_command(
    "rm -rf /root/project/data/raw/crypto && "
    "mkdir -p /root/project/data/raw && "
    "cd /root/project/data/raw && "
    "tar xzf /root/project/crypto_data.tar.gz && "
    "rm /root/project/crypto_data.tar.gz && "
    "echo FILES: && find crypto -name '*.parquet' | wc -l && "
    "echo SIZE: && du -sh crypto/",
    timeout=120,
)
out = stdout.read().decode()
err = stderr.read().decode()
print(out, flush=True)
if err.strip():
    print(f"[stderr] {err}", flush=True)
ssh.close()

# Clean up local temp
os.remove(tar_path)
print("\nAll done!", flush=True)
