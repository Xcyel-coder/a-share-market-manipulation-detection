import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Get the error traceback
stdin, stdout, stderr = ssh.exec_command(
    "grep -A 20 'Traceback' /root/project/outputs/comprehensive/experiment.log 2>/dev/null | tail -30", timeout=10)
print("=== ERROR DETAIL ===")
print(stdout.read().decode())

# Get last 15 lines
stdin, stdout, stderr = ssh.exec_command(
    "tail -15 /root/project/outputs/comprehensive/experiment.log 2>/dev/null", timeout=10)
print("=== LOG TAIL ===")
print(stdout.read().decode())

# Check if results JSON was saved (partial)
stdin, stdout, stderr = ssh.exec_command(
    "cat /root/project/outputs/comprehensive/comprehensive_results.json 2>/dev/null | head -20", timeout=10)
print("=== SAVED RESULTS ===")
print(stdout.read().decode())

ssh.close()
