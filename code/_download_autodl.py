"""
Download all important files from AutoDL to local.
Strategy: (1) tar small files, (2) download weight files individually.
"""
import paramiko
import os
from pathlib import Path
import time

HOST = "connect.westd.seetacloud.com"
PORT = 26391
USER = "root"
PASSWD = os.environ["AUTODL_SSH_PASSWORD"]

LOCAL = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\autodl_backup")
LOCAL.mkdir(exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to {HOST}:{PORT}...")
ssh.connect(HOST, port=PORT, username=USER, password=PASSWD, timeout=30)
sftp = ssh.open_sftp()
print("Connected!")

def run_cmd(cmd, timeout=120):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace").strip()

def download_file(remote, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        info = sftp.stat(remote)
        size_mb = info.st_size / (1024 * 1024)
        print(f"  [{size_mb:.1f}MB] {remote} -> {local_path.name}")
        t0 = time.time()
        sftp.get(remote, str(local_path))
        elapsed = time.time() - t0
        speed = size_mb / max(elapsed, 0.1)
        print(f"    Done in {elapsed:.1f}s ({speed:.1f} MB/s)")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

# ========== PHASE 1: Pack & download small files ==========
print("\n===== PHASE 1: Scripts, configs, JSON results, logs =====")
print("Creating tar archive on remote (excluding .pt files)...")
run_cmd(
    "cd /root/project && tar czf /tmp/project_small.tar.gz "
    "--exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' "
    "--exclude='data/raw/crypto_data' --exclude='data/raw/astock_daily' "
    ".",
    timeout=120
)
tar_size = run_cmd("ls -lh /tmp/project_small.tar.gz | awk '{print $5}'")
print(f"Archive size: {tar_size}")

local_tar = LOCAL / "project_small.tar.gz"
download_file("/tmp/project_small.tar.gz", local_tar)

# ========== PHASE 2: Pretrain best weights (3 files) ==========
print("\n===== PHASE 2: Pretrain best weights (3 x ~98MB) =====")
pretrain_files = [
    "/root/project/outputs/pretrain/round1_validation/best.pt",
    "/root/project/outputs/pretrain/round2_tuning/best.pt",
    "/root/project/outputs/pretrain/round3_full/best.pt",
]
for remote in pretrain_files:
    parts = remote.split("/outputs/")[1]
    local_path = LOCAL / "weights" / parts
    download_file(remote, local_path)

# ========== PHASE 3: Clean rerun weights (final results, 20 files) ==========
print("\n===== PHASE 3: Clean rerun weights (L0-L3 x 5 folds, 20 x ~98MB) =====")
for level in ["L0", "L1", "L2", "L3"]:
    for fold in range(5):
        remote = f"/root/project/outputs/clean_rerun/{level}/fold_{fold}/best.pt"
        local_path = LOCAL / "weights" / "clean_rerun" / level / f"fold_{fold}" / "best.pt"
        download_file(remote, local_path)

# ========== PHASE 4: Pretrain final.pt (round3) ==========
print("\n===== PHASE 4: Pretrain round3 final.pt =====")
download_file(
    "/root/project/outputs/pretrain/round3_full/final.pt",
    LOCAL / "weights" / "pretrain" / "round3_full" / "final.pt"
)

# ========== PHASE 5: Data directory (exclude large crypto raw) ==========
print("\n===== PHASE 5: Data directory (processed data, CoT embeddings) =====")
run_cmd(
    "cd /root/project && tar czf /tmp/project_data.tar.gz "
    "--exclude='data/raw/crypto_data' "
    "data/",
    timeout=120
)
data_size = run_cmd("ls -lh /tmp/project_data.tar.gz | awk '{print $5}'")
print(f"Data archive size: {data_size}")
download_file("/tmp/project_data.tar.gz", LOCAL / "project_data.tar.gz")

# ========== Summary ==========
print("\n===== DOWNLOAD SUMMARY =====")
total = 0
for f in LOCAL.rglob("*"):
    if f.is_file():
        total += f.stat().st_size
print(f"Total downloaded: {total / (1024**3):.2f} GB")
print(f"Saved to: {LOCAL}")

# List all downloaded files
for f in sorted(LOCAL.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.relative_to(LOCAL)} ({size_mb:.1f} MB)")

run_cmd("rm -f /tmp/project_small.tar.gz /tmp/project_data.tar.gz")
sftp.close()
ssh.close()
print("\nAll done!")
