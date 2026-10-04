import os
# -*- coding: utf-8 -*-
import paramiko, sys, json
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Get the comprehensive results JSON
stdin, stdout, stderr = ssh.exec_command(
    "cat /root/project/outputs/comprehensive/comprehensive_results.json 2>/dev/null", timeout=10)
data = stdout.read().decode()
print("=== COMPREHENSIVE RESULTS JSON ===")
print(data)

# Also get the last 30 lines of the log for the final summary
stdin, stdout, stderr = ssh.exec_command(
    "tail -30 /root/project/outputs/comprehensive/experiment_part2.log 2>/dev/null", timeout=10)
print("\n=== FINAL LOG ===")
print(stdout.read().decode())

ssh.close()
