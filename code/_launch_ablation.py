import os
# -*- coding: utf-8 -*-
import paramiko, sys, time
sys.stdout.reconfigure(encoding='utf-8')

PY = "/root/miniconda3/bin/python"
LOCAL = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\run_ablation_study.py"
REMOTE = "/root/project/run_ablation_study.py"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

sftp = ssh.open_sftp()
sftp.put(LOCAL, REMOTE)
sftp.close()
print("Script uploaded.", flush=True)

launcher = f"""#!/bin/bash
cd /root/project && {PY} {REMOTE} > /root/project/ablation_stdout.log 2>&1 &
echo $! > /tmp/ablation_pid.txt
echo "PID: $(cat /tmp/ablation_pid.txt)"
"""
sftp = ssh.open_sftp()
with sftp.open("/root/project/launch_ablation.sh", "w") as f:
    f.write(launcher)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash /root/project/launch_ablation.sh", timeout=10)
print(stdout.read().decode().strip(), flush=True)

time.sleep(15)

stdin, stdout, stderr = ssh.exec_command("tail -20 /root/project/ablation_stdout.log 2>/dev/null", timeout=10)
print("\nInitial log:", flush=True)
print(stdout.read().decode().strip(), flush=True)

ssh.close()
print("\nAblation launched.", flush=True)
