import os
# -*- coding: utf-8 -*-
import paramiko, sys, json
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Ablation results
stdin, stdout, stderr = ssh.exec_command("cat /root/project/outputs/ablation_study/ablation_results.json 2>/dev/null", timeout=10)
print("=== ABLATION RESULTS ===")
print(stdout.read().decode())

# Expanded finetune results
stdin, stdout, stderr = ssh.exec_command("cat /root/project/outputs/expanded_finetune/all_results.json 2>/dev/null", timeout=10)
print("=== EXPANDED FINETUNE (126 cases + daily pretrain) ===")
print(stdout.read().decode())

# Original comparison
stdin, stdout, stderr = ssh.exec_command("""
/root/miniconda3/bin/python -c "
import json
with open('/root/project/outputs/clean_rerun/comparison_results.json') as f:
    d = json.load(f)
for k,v in d.items():
    if 'mcc' in v:
        print(f'{k}: MCC={v[\"mcc\"]:.4f}±{v.get(\"mcc_std\",0):.4f}')
" 2>/dev/null""", timeout=10)
print("=== ORIGINAL 23-CASE RESULTS ===")
print(stdout.read().decode())

# Ablation results
stdin, stdout, stderr = ssh.exec_command("""
/root/miniconda3/bin/python -c "
import json
with open('/root/project/outputs/clean_rerun/ablation_results.json') as f:
    d = json.load(f)
for k,v in d.items():
    if 'mcc' in v:
        print(f'{k}: MCC={v[\"mcc\"]:.4f}±{v.get(\"mcc_std\",0):.4f}')
" 2>/dev/null""", timeout=10)
print("=== ORIGINAL ABLATION (w/o pretrain etc.) ===")
print(stdout.read().decode())

ssh.close()
