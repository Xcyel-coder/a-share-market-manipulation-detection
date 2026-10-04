import os
# -*- coding: utf-8 -*-
import paramiko, sys, json
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

cmds = [
    'echo "=== CASES JSON ===" && find /root/project -name "cases.json" 2>/dev/null',
    'echo "=== METADATA FILES ===" && find /root/project/data/raw/astock -name "*.json" 2>/dev/null | head -5',
    'echo "=== ASTOCK DIR LISTING ===" && ls /root/project/data/raw/astock/ 2>/dev/null',
    'echo "=== SAMPLE CASE DIR ===" && ls -la /root/project/data/raw/astock/CSRC-2017-064/ 2>/dev/null',
    'echo "=== SAMPLE METADATA ===" && cat /root/project/data/raw/astock/CSRC-2017-064/meta.json 2>/dev/null',
    'echo "=== SAMPLE PARQUET COLS ===" && /root/miniconda3/bin/python -c "import pandas as pd; df=pd.read_parquet(\'/root/project/data/raw/astock/CSRC-2017-064/daily.parquet\'); print(list(df.columns)); print(df.head(2).to_string()); print(f\'Shape: {df.shape}\')" 2>&1',
    'echo "=== CASES LIST ===" && find /root/project -name "cases.json" -exec cat {} \\; 2>/dev/null | head -80',
    'echo "=== FINETUNE CONFIG ===" && cat /root/project/model/configs/finetune_config.yaml 2>/dev/null',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    print(stdout.read().decode())
    err = stderr.read().decode().strip()
    if err and 'Warning' not in err:
        print(f"[ERR] {err[:200]}")

ssh.close()
