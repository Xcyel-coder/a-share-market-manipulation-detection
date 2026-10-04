import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

cmds = [
    'echo "=== PRETRAIN WEIGHTS ===" && find /root/project/outputs/pretrain -type f 2>/dev/null | head -30',
    'echo "=== PYTHON VERSION ===" && which python && python --version',
    'echo "=== CRYPTO DATA SIZE ===" && du -sh /root/project/data/raw/crypto/*/ 2>/dev/null',
    'echo "=== SAMPLE CRYPTO PARQUET ===" && python -c "import pandas as pd; df=pd.read_parquet(\'/root/project/data/raw/crypto/BTCUSDT/2021.parquet\'); print(f\'Shape: {df.shape}\'); print(f\'Columns: {list(df.columns)}\'); print(df.head(3).to_string())" 2>&1',
    'echo "=== PRETRAIN CONFIG ===" && cat /root/project/model/configs/pretrain_config.yaml 2>/dev/null',
    'echo "=== CLEAN RERUN OUTPUTS ===" && ls -la /root/project/outputs/clean_rerun/ 2>/dev/null',
    'echo "=== DISK DETAIL ===" && du -sh /root/project/data/ /root/project/outputs/ /root/project/model/ 2>/dev/null',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print(out)
    if err.strip() and 'WARNING' not in err:
        print(f"[STDERR] {err[:300]}")

ssh.close()
print("Done.")
