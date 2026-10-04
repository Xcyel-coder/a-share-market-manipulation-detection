# -*- coding: utf-8 -*-
"""Upload new files to server and start scale-aligned pre-training."""
import paramiko
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

LOCAL_BASE = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\model"
REMOTE_BASE = "/root/project/model"
PY = "/root/miniconda3/bin/python"

FILES_TO_UPLOAD = [
    ("data/crypto_daily_dataset.py", "data/crypto_daily_dataset.py"),
    ("configs/pretrain_daily_config.yaml", "configs/pretrain_daily_config.yaml"),
    ("scripts/run_pretrain_daily.py", "scripts/run_pretrain_daily.py"),
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)
sftp = ssh.open_sftp()

# Upload files
for local_rel, remote_rel in FILES_TO_UPLOAD:
    local_path = os.path.join(LOCAL_BASE, local_rel)
    remote_path = f"{REMOTE_BASE}/{remote_rel}"
    print(f"Uploading: {local_rel} -> {remote_path}")
    sftp.put(local_path, remote_path)
    print(f"  OK ({os.path.getsize(local_path)} bytes)")

# Also make sure astock_dataset.py is up to date (has compute_daily_features)
local_astock = os.path.join(LOCAL_BASE, "data/astock_dataset.py")
remote_astock = f"{REMOTE_BASE}/data/astock_dataset.py"
print(f"Uploading: data/astock_dataset.py -> {remote_astock}")
sftp.put(local_astock, remote_astock)
print(f"  OK ({os.path.getsize(local_astock)} bytes)")

sftp.close()

# Quick verification: check n_features from daily data
print("\n--- Verifying daily data can be loaded ---")
verify_cmd = f"""cd /root/project/model && {PY} -c "
import sys
sys.path.insert(0, '.')
from data.crypto_daily_dataset import CryptoDailyPretrainDataset

ds = CryptoDailyPretrainDataset(
    data_dir='/root/project/data/processed/crypto_aggregated',
    symbols=['BTCUSDT'],
    timeframe='daily',
    seq_len=60,
    stride=10,
)
print(f'Windows: {{len(ds)}}')
print(f'Features: {{ds.n_features}}')
print(f'Feature names: {{ds.feature_names}}')
print(f'Anomaly ratio: {{ds.get_anomaly_ratio():.3f}}')
sample = ds[0]
print(f'Sample shapes: features={{sample[\"features\"].shape}}, mask={{sample[\"mask\"].shape}}')

ds4h = CryptoDailyPretrainDataset(
    data_dir='/root/project/data/processed/crypto_aggregated',
    symbols=['BTCUSDT'],
    timeframe='4h',
    seq_len=60,
    stride=10,
)
print(f'4H windows: {{len(ds4h)}}')
print(f'4H features: {{ds4h.n_features}}')
print(f'4H anomaly ratio: {{ds4h.get_anomaly_ratio():.3f}}')
" 2>&1"""

stdin, stdout, stderr = ssh.exec_command(verify_cmd, timeout=60)
print(stdout.read().decode())
err = stderr.read().decode()
if err.strip():
    print(f"[STDERR] {err[:500]}")

ssh.close()
print("Upload and verification complete.")
