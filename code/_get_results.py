import os
# -*- coding: utf-8 -*-
import paramiko, sys, json
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Get full results
stdin, stdout, stderr = ssh.exec_command("cat /root/project/outputs/expanded_finetune/all_results.json 2>/dev/null", timeout=10)
print("=== EXPANDED FINETUNE RESULTS ===")
print(stdout.read().decode())

# Get original experiment results for comparison
stdin, stdout, stderr = ssh.exec_command("cat /root/project/outputs/clean_rerun/all_results.json 2>/dev/null | /root/miniconda3/bin/python -c \"import sys,json; d=json.load(sys.stdin); [print(f'{k}: MCC={v[\"avg_mcc\"]:.4f}±{v[\"std_mcc\"]:.4f}') for k,v in d.items() if 'avg_mcc' in v]\" 2>/dev/null", timeout=10)
print("=== ORIGINAL EXPERIMENT (23 cases, minute pretrain) ===")
print(stdout.read().decode())

# Get pretrain daily summary
stdin, stdout, stderr = ssh.exec_command("cat /root/project/outputs/pretrain_daily/pretrain_daily_summary.json 2>/dev/null", timeout=10)
print("=== PRETRAIN DAILY SUMMARY ===")
print(stdout.read().decode())

# Get original pretrain summary
stdin, stdout, stderr = ssh.exec_command("cat /root/project/outputs/pretrain/pretrain_summary.json 2>/dev/null", timeout=10)
print("=== ORIGINAL PRETRAIN SUMMARY ===")
print(stdout.read().decode())

ssh.close()
