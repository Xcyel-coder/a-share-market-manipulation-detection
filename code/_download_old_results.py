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

sftp = ssh.open_sftp()
base = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\outputs"
os.makedirs(base, exist_ok=True)

# List all result files on server
for d in ["comprehensive", "quality_weighted"]:
    path = f"/root/project/outputs/{d}"
    try:
        files = sftp.listdir(path)
        print(f"\n--- {path} ---", flush=True)
        for f in files:
            print(f"  {f}", flush=True)
    except Exception as e:
        print(f"  {d}: {e}", flush=True)

# Download comprehensive results
try:
    sftp.get("/root/project/outputs/comprehensive/comprehensive_results.json",
             os.path.join(base, "comprehensive_results.json"))
    print("\nDownloaded: comprehensive_results.json", flush=True)
except Exception as e:
    print(f"Failed comprehensive: {e}", flush=True)

# Also get the experiment log summary
try:
    _, stdout, _ = ssh.exec_command(
        "grep -E '(COMPREHENSIVE RESULTS|Method|MoE-|RandomForest|XGBoost|SVM|LSTM|Transformer|====)' "
        "/root/project/outputs/comprehensive/experiment.log | tail -20"
    )
    print("\n--- Old experiment summary ---", flush=True)
    print(stdout.read().decode().strip(), flush=True)
except Exception as e:
    print(f"Log grep failed: {e}", flush=True)

sftp.close()
ssh.close()
print("\nDone", flush=True)
