import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

cmds = [
    'echo "=== CASES DIR ===" && ls -la /root/project/data/processed/extracted_cases/ 2>/dev/null',
    'echo "=== CASES FILES ===" && find /root/project/data/processed -name "*.json" 2>/dev/null | head -10',
    'echo "=== CASES CONTENT ===" && cat /root/project/data/processed/extracted_cases/cases.json 2>/dev/null | head -100',
    'echo "=== RUN SCRIPTS ===" && ls /root/project/run_*.py 2>/dev/null',
    'echo "=== FINETUNE SCRIPT HEAD ===" && head -80 /root/project/model/scripts/run_finetune.py 2>/dev/null',
    'echo "=== TRAINING PROGRESS ===" && tail -15 /root/project/pretrain_daily_stdout.log 2>/dev/null',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    print(stdout.read().decode())

ssh.close()
