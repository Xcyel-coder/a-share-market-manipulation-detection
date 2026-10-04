import os
# -*- coding: utf-8 -*-
"""Kill duplicate processes and monitor training."""
import paramiko
import sys
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Kill duplicate processes from first failed attempt (PID 2084, 2085)
print("--- Killing duplicate processes ---")
stdin, stdout, stderr = ssh.exec_command("kill -9 2084 2085 2>/dev/null; echo done", timeout=5)
print(stdout.read().decode().strip())

# Check remaining processes
import time
time.sleep(2)
stdin, stdout, stderr = ssh.exec_command("ps aux | grep pretrain_daily | grep -v grep", timeout=5)
print("\nRemaining processes:")
print(stdout.read().decode().strip())

# Check training progress
stdin, stdout, stderr = ssh.exec_command("tail -30 /root/project/pretrain_daily_stdout.log 2>/dev/null", timeout=10)
print("\nLatest log:")
print(stdout.read().decode().strip())

# Check GPU usage
stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null", timeout=10)
print("\nGPU status:")
print(stdout.read().decode().strip())

ssh.close()
