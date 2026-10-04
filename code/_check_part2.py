import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null', timeout=5)
print(f'GPU: {stdout.read().decode().strip()}')

stdin, stdout, stderr = ssh.exec_command('ps aux | grep part2 | grep -v grep | head -1', timeout=5)
ps = stdout.read().decode().strip()
print(f'Process: {"RUNNING" if ps else "NOT RUNNING"}')

stdin, stdout, stderr = ssh.exec_command(
    r"grep -E '(MoE-L3|best MCC|Early stop|BASELINE|RandomForest|XGBoost|SVM|LSTM|Transformer|====|MCC=)' "
    "/root/project/outputs/comprehensive/experiment_part2.log 2>/dev/null | tail -40", timeout=10)
print(f'Key lines:\n{stdout.read().decode()}')

stdin, stdout, stderr = ssh.exec_command(
    "grep -i 'error\\|traceback' /root/project/outputs/comprehensive/experiment_part2.log 2>/dev/null | tail -3", timeout=10)
err = stdout.read().decode().strip()
if err:
    print(f'ERRORS:\n{err}')
else:
    print('No errors.')

stdin, stdout, stderr = ssh.exec_command('wc -l /root/project/outputs/comprehensive/experiment_part2.log 2>/dev/null', timeout=5)
print(f'Log lines: {stdout.read().decode().strip()}')

ssh.close()
