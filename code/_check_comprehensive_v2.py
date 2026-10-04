import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null', timeout=5)
print(f'GPU: {stdout.read().decode().strip()}')

stdin, stdout, stderr = ssh.exec_command('ps aux | grep comprehensive | grep -v grep | head -1', timeout=5)
ps = stdout.read().decode().strip()
print(f'Process: {"RUNNING" if ps else "NOT RUNNING"}')

# Get summary lines (MCC averages, level transitions)
stdin, stdout, stderr = ssh.exec_command(
    r"grep -E '(MoE-L|best MCC|Early stop|BASELINE|MCC =|RandomForest|XGBoost|SVM|LSTM|Transformer|============)' "
    "/root/project/outputs/comprehensive/experiment.log 2>/dev/null | tail -60", timeout=10)
print(f'Key log lines:\n{stdout.read().decode()}')

stdin, stdout, stderr = ssh.exec_command('wc -l /root/project/outputs/comprehensive/experiment.log 2>/dev/null', timeout=5)
print(f'Total log lines: {stdout.read().decode().strip()}')

# Check for any errors
stdin, stdout, stderr = ssh.exec_command(
    "grep -i 'error\\|traceback\\|exception' /root/project/outputs/comprehensive/experiment.log 2>/dev/null | tail -5", timeout=10)
err = stdout.read().decode().strip()
if err:
    print(f'ERRORS:\n{err}')

ssh.close()
