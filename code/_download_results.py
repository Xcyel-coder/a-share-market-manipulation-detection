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

base = r"c:\Users\wangm\OneDrive - Audencia\毕业论文"
files = [
    ("/root/project/outputs/quality_weighted/quality_weighted_results.json",
     os.path.join(base, "outputs", "quality_weighted_results.json")),
    ("/root/project/outputs/quality_weighted/experiment.log",
     os.path.join(base, "outputs", "quality_weighted_experiment.log")),
]

os.makedirs(os.path.join(base, "outputs"), exist_ok=True)

for remote, local in files:
    try:
        sftp.get(remote, local)
        print(f"Downloaded: {os.path.basename(local)}", flush=True)
    except Exception as e:
        print(f"Failed: {remote} -> {e}", flush=True)

sftp.close()
ssh.close()
print("Done", flush=True)
