import os
# -*- coding: utf-8 -*-
"""Upload and launch expanded fine-tuning on server."""
import paramiko
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

PY = "/root/miniconda3/bin/python"
LOCAL_SCRIPT = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\run_expanded_finetune.py"
REMOTE_SCRIPT = "/root/project/run_expanded_finetune.py"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

sftp = ssh.open_sftp()
sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
sftp.close()
print("Script uploaded.", flush=True)

# Launch
launcher = f"""#!/bin/bash
cd /root/project && {PY} {REMOTE_SCRIPT} > /root/project/expanded_finetune_stdout.log 2>&1 &
echo $! > /tmp/expanded_finetune_pid.txt
echo "PID: $(cat /tmp/expanded_finetune_pid.txt)"
"""
sftp = ssh.open_sftp()
with sftp.open("/root/project/launch_finetune.sh", "w") as f:
    f.write(launcher)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash /root/project/launch_finetune.sh", timeout=10)
print(stdout.read().decode().strip(), flush=True)

time.sleep(10)

stdin, stdout, stderr = ssh.exec_command("cat /tmp/expanded_finetune_pid.txt 2>/dev/null && echo '---' && ps aux | grep expanded_finetune | grep -v grep | head -3", timeout=10)
print("Process check:", flush=True)
print(stdout.read().decode().strip(), flush=True)

stdin, stdout, stderr = ssh.exec_command("head -30 /root/project/expanded_finetune_stdout.log 2>/dev/null", timeout=10)
print("\nInitial log:", flush=True)
print(stdout.read().decode().strip(), flush=True)

ssh.close()
print("\nFine-tuning launched.", flush=True)
