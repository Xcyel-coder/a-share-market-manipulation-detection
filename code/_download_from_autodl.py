"""
Connect to AutoDL and download all important project files to local backup.
"""
import paramiko
import os
import stat
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 26391
USER = "root"
PASSWD = os.environ["AUTODL_SSH_PASSWORD"]

LOCAL_BASE = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\autodl_backup")
LOCAL_BASE.mkdir(exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to {HOST}:{PORT}...")
ssh.connect(HOST, port=PORT, username=USER, password=PASSWD, timeout=30)
sftp = ssh.open_sftp()
print("Connected!\n")

def run(cmd):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    return stdout.read().decode("utf-8", errors="replace").strip()

print("=== Outputs directory tree ===")
print(run("find /root/project/outputs -type f | head -80"))

print("\n=== Data directory tree ===")
print(run("find /root/project/data -type f 2>/dev/null"))

print("\n=== Model code directory ===")
print(run("find /root/project/model -type f -name '*.py' -o -name '*.json' -o -name '*.yaml' 2>/dev/null"))

print("\n=== Size of key directories ===")
print(run("du -sh /root/project/outputs/* /root/project/data /root/project/model 2>/dev/null"))

print("\n=== Size of outputs subdirs ===")
print(run("du -sh /root/project/outputs/clean_rerun/* 2>/dev/null"))

print("\n=== JSON files in outputs ===")
print(run("find /root/project/outputs -name '*.json' 2>/dev/null"))

print("\n=== Pretrain outputs ===")
print(run("find /root/project/outputs/pretrain -type f 2>/dev/null"))
print(run("find /root/project/model/outputs -type f 2>/dev/null"))

downloaded = 0
skipped = 0

def download_recursive(remote_dir, local_dir):
    global downloaded, skipped
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = sftp.listdir_attr(remote_dir)
    except IOError:
        print(f"  [SKIP] Cannot list {remote_dir}")
        return

    for entry in entries:
        remote_path = f"{remote_dir}/{entry.filename}"
        local_path = local_dir / entry.filename

        if stat.S_ISDIR(entry.st_mode):
            download_recursive(remote_path, local_path)
        else:
            size_mb = entry.st_size / (1024 * 1024)
            if local_path.exists() and local_path.stat().st_size == entry.st_size:
                skipped += 1
                continue
            print(f"  DL: {remote_path} ({size_mb:.1f} MB)")
            try:
                sftp.get(remote_path, str(local_path))
                downloaded += 1
            except Exception as e:
                print(f"  [ERR] {e}")

targets = [
    ("/root/project/outputs", LOCAL_BASE / "outputs"),
    ("/root/project/data", LOCAL_BASE / "data"),
    ("/root/project/model", LOCAL_BASE / "model"),
]

root_files = [
    "baselines_v2_stdout.log",
    "clean_rerun_stdout.log",
    "cv_robustness_stdout.log",
    "finetune_stdout.log",
    "l1l3_stdout.log",
    "phase4_stdout.log",
    "rerun_stdout.log",
    "run_backtest_fix.py",
    "run_baselines_v2.py",
    "run_clean_rerun.py",
    "run_cv_robustness.py",
    "run_finetune_remote.py",
    "run_l1l3.py",
    "run_phase4.py",
    "run_rerun.py",
    "setup_remote.sh",
]

print("\n" + "=" * 60)
print("DOWNLOADING FILES")
print("=" * 60)

for remote_dir, local_dir in targets:
    print(f"\n--- {remote_dir} -> {local_dir} ---")
    download_recursive(remote_dir, local_dir)

scripts_dir = LOCAL_BASE / "scripts"
scripts_dir.mkdir(exist_ok=True)
for fname in root_files:
    remote_path = f"/root/project/{fname}"
    local_path = scripts_dir / fname
    try:
        remote_stat = sftp.stat(remote_path)
        if local_path.exists() and local_path.stat().st_size == remote_stat.st_size:
            skipped += 1
            continue
        size_mb = remote_stat.st_size / (1024 * 1024)
        print(f"  DL: {remote_path} ({size_mb:.2f} MB)")
        sftp.get(remote_path, str(local_path))
        downloaded += 1
    except Exception as e:
        print(f"  [SKIP] {fname}: {e}")

sftp.close()
ssh.close()

print(f"\n{'=' * 60}")
print(f"DONE: {downloaded} files downloaded, {skipped} skipped (already exist)")
print(f"Saved to: {LOCAL_BASE}")
