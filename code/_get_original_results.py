import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Find all result JSONs
stdin, stdout, stderr = ssh.exec_command("find /root/project/outputs -name '*.json' -type f 2>/dev/null", timeout=10)
print("=== ALL JSON FILES ===")
print(stdout.read().decode())

# Try reading different result files
for path in [
    "/root/project/outputs/clean_rerun/comparison_results.json",
    "/root/project/outputs/clean_rerun/ablation_results.json",
    "/root/project/outputs/finetune/results.json",
    "/root/project/outputs/finetune/all_results.json",
]:
    stdin, stdout, stderr = ssh.exec_command(f"cat {path} 2>/dev/null | head -50", timeout=10)
    content = stdout.read().decode().strip()
    if content:
        print(f"\n=== {path} ===")
        print(content)

# Also check the original experiment logs
stdin, stdout, stderr = ssh.exec_command("ls -la /root/project/outputs/clean_rerun/ 2>/dev/null", timeout=10)
print("\n=== clean_rerun dir ===")
print(stdout.read().decode())

stdin, stdout, stderr = ssh.exec_command("tail -30 /root/project/outputs/clean_rerun/*.log 2>/dev/null | head -80", timeout=10)
print("\n=== clean_rerun logs tail ===")
print(stdout.read().decode())

ssh.close()
