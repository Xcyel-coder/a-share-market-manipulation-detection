import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

cmds = [
    'echo "=== PROCESS ===" && ps aux | grep ablation_study | grep -v grep | head -3',
    'echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null',
    'echo "=== LOG ===" && tail -30 /root/project/ablation_stdout.log 2>/dev/null',
]
for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
    print(stdout.read().decode(), flush=True)

ssh.close()
