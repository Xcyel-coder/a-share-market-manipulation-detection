import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null', timeout=5)
print(f'GPU: {stdout.read().decode().strip()}')

stdin, stdout, stderr = ssh.exec_command('ps aux | grep comprehensive | grep -v grep', timeout=5)
ps = stdout.read().decode().strip()
print(f'Process: {ps if ps else "NOT RUNNING"}')

stdin, stdout, stderr = ssh.exec_command('tail -30 /root/project/outputs/comprehensive/experiment.log 2>/dev/null', timeout=10)
print(f'Log:\n{stdout.read().decode()}')

stdin, stdout, stderr = ssh.exec_command('wc -l /root/project/outputs/comprehensive/experiment.log 2>/dev/null', timeout=5)
print(f'Log lines: {stdout.read().decode().strip()}')

ssh.close()
