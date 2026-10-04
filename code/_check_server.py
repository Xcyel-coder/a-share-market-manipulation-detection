import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

cmds = [
    'echo "=== DISK ===" && df -h / /root',
    'echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv',
    'echo "=== PYTHON ===" && python3 --version && pip list 2>/dev/null | grep -iE "torch|numpy|pandas|paramiko"',
    'echo "=== PROJECT DIR ===" && ls -la /root/project/ 2>/dev/null || echo "No /root/project"',
    'echo "=== PROJECT SUBDIRS ===" && find /root/project -maxdepth 2 -type d 2>/dev/null | head -30',
    'echo "=== CRYPTO DATA ===" && find /root -path "*/crypto*" -name "*.parquet" 2>/dev/null | head -20',
    'echo "=== PRETRAIN OUTPUTS ===" && find /root -name "best.pt" 2>/dev/null | head -10',
    'echo "=== ASTOCK DATA ===" && find /root -path "*astock*" -name "*.parquet" 2>/dev/null | head -20',
    'echo "=== CONFIG FILES ===" && find /root/project -name "*.yaml" 2>/dev/null | head -10',
    'echo "=== WEIGHTS ===" && find /root -name "*.pt" 2>/dev/null | head -20',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print(out)
    if err.strip():
        print(f"[STDERR] {err[:200]}")

ssh.close()
print("Done.")
