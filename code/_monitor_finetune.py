import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

cmds = [
    'echo "=== PROCESSES ===" && ps aux | grep -E "expanded_finetune|pretrain_daily" | grep -v grep',
    'echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null',
    'echo "=== FINETUNE LOG TAIL ===" && tail -60 /root/project/expanded_finetune_stdout.log 2>/dev/null',
    'echo "=== OUTPUT FILES ===" && find /root/project/outputs/expanded_finetune -type f -name "*.pt" -o -name "*.json" 2>/dev/null | sort',
    'echo "=== PRETRAIN DAILY ===" && ls -la /root/project/outputs/pretrain_daily/*/best.pt 2>/dev/null',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    print(stdout.read().decode(), flush=True)

ssh.close()
