"""Upload tar.gz with progress tracking, then extract on remote."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

tar_path = os.path.join(os.environ["TEMP"], "crypto_data.tar.gz")
if not os.path.exists(tar_path):
    print("ERROR: tar.gz not found. Creating it first...")
    import tarfile
    local_crypto = Path(__file__).parent / "data_pipeline" / "data" / "raw" / "crypto"
    with tarfile.open(tar_path, "w:gz") as tar:
        for pf in local_crypto.rglob("*.parquet"):
            arcname = str(pf.relative_to(local_crypto.parent.parent))
            tar.add(str(pf), arcname=arcname)

file_size = os.path.getsize(tar_path)
print(f"Uploading {tar_path} ({file_size / 1024 / 1024:.1f} MB)")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
sftp = ssh.open_sftp()

last_pct = [0]
def progress(transferred, total):
    pct = int(transferred / total * 100)
    if pct >= last_pct[0] + 10:
        last_pct[0] = pct
        print(f"  Upload: {pct}% ({transferred // 1024 // 1024} / {total // 1024 // 1024} MB)", flush=True)

print("Starting upload...", flush=True)
sftp.put(tar_path, "/root/project/crypto_data.tar.gz", callback=progress)
print("Upload complete!", flush=True)
sftp.close()

print("Extracting on remote...", flush=True)
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/project/data/raw && "
    "rm -rf crypto && "
    "tar xzf /root/project/crypto_data.tar.gz && "
    "rm /root/project/crypto_data.tar.gz && "
    "echo EXTRACTED && "
    "find crypto -name '*.parquet' | wc -l && "
    "du -sh crypto/",
    timeout=120,
)
print(stdout.read().decode(), flush=True)
err = stderr.read().decode()
if err.strip():
    print(f"[stderr] {err}", flush=True)
ssh.close()

os.remove(tar_path)
print("All done!", flush=True)
