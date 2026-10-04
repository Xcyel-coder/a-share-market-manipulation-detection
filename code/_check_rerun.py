import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

cmds = [
    'head -100 /root/project/run_clean_rerun.py 2>/dev/null',
    'echo "=== SAMPLE EXTRACTED ===" && cat /root/project/data/processed/extracted_cases/CSRC-2017-064_extracted.json 2>/dev/null',
    'echo "=== EXTRACTION SUMMARY ===" && cat /root/project/data/processed/extracted_cases/extraction_summary.json 2>/dev/null | head -60',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
    print(stdout.read().decode())

ssh.close()
