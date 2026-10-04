import os
# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUNBUFFERED"] = "1"
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.westd.seetacloud.com", port=28784,
            username="root", password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Check process
_, stdout, _ = ssh.exec_command("ps aux | grep quality_weighted | grep -v grep")
ps = stdout.read().decode().strip()
if ps:
    print(f"进程运行中: {ps[:100]}", flush=True)
else:
    print("进程已结束或未运行", flush=True)

# GPU usage
_, stdout, _ = ssh.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader")
print(f"GPU: {stdout.read().decode().strip()}", flush=True)

# Tail log
_, stdout, _ = ssh.exec_command("tail -40 /root/project/outputs/quality_weighted/run.log 2>/dev/null")
log = stdout.read().decode().strip()
print(f"\n--- 最新日志 ---\n{log}", flush=True)

# Check if results file exists
_, stdout, _ = ssh.exec_command("ls -la /root/project/outputs/quality_weighted/quality_weighted_results.json 2>/dev/null")
res = stdout.read().decode().strip()
if res:
    print(f"\n结果文件已生成: {res}", flush=True)

ssh.close()
