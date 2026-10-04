import os
# -*- coding: utf-8 -*-
"""Kill old process, re-upload and relaunch."""
import paramiko, sys, time
sys.stdout.reconfigure(encoding='utf-8')

PY = "/root/miniconda3/bin/python"
LOCAL = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\run_expanded_finetune.py"
REMOTE = "/root/project/run_expanded_finetune.py"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Kill old
stdin, stdout, stderr = ssh.exec_command("cat /tmp/expanded_finetune_pid.txt 2>/dev/null", timeout=5)
old_pid = stdout.read().decode().strip()
if old_pid:
    ssh.exec_command(f"kill -9 {old_pid} 2>/dev/null", timeout=5)
    time.sleep(1)
    print(f"Killed old PID: {old_pid}", flush=True)

# Re-upload
sftp = ssh.open_sftp()
sftp.put(LOCAL, REMOTE)
sftp.close()
print("Script re-uploaded.", flush=True)

# Clean old output dir to avoid stale results
ssh.exec_command("rm -rf /root/project/outputs/expanded_finetune", timeout=5)
time.sleep(1)

# Launch
launcher = f"""#!/bin/bash
cd /root/project && {PY} {REMOTE} > /root/project/expanded_finetune_stdout.log 2>&1 &
echo $! > /tmp/expanded_finetune_pid.txt
echo "PID: $(cat /tmp/expanded_finetune_pid.txt)"
"""
sftp = ssh.open_sftp()
with sftp.open("/root/project/launch_finetune.sh", "w") as f:
    f.write(launcher)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash /root/project/launch_finetune.sh", timeout=10)
print(stdout.read().decode().strip(), flush=True)

time.sleep(15)

stdin, stdout, stderr = ssh.exec_command("tail -20 /root/project/expanded_finetune_stdout.log 2>/dev/null", timeout=10)
print("\nInitial log:", flush=True)
print(stdout.read().decode().strip(), flush=True)

ssh.close()
print("\nDone.", flush=True)
