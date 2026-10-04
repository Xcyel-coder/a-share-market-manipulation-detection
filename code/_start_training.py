import os
# -*- coding: utf-8 -*-
"""Start scale-aligned pre-training on the server."""
import paramiko
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

PY = "/root/miniconda3/bin/python"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Write a launcher script on the server
launcher = """#!/bin/bash
cd /root/project/model
{py} -m scripts.run_pretrain_daily \\
    --config configs/pretrain_daily_config.yaml \\
    --round all \\
    > /root/project/pretrain_daily_stdout.log 2>&1 &
echo $! > /tmp/pretrain_daily_pid.txt
echo "PID: $(cat /tmp/pretrain_daily_pid.txt)"
""".format(py=PY)

sftp = ssh.open_sftp()
with sftp.open("/root/project/launch_pretrain_daily.sh", "w") as f:
    f.write(launcher)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("chmod +x /root/project/launch_pretrain_daily.sh && bash /root/project/launch_pretrain_daily.sh", timeout=10)
out = stdout.read().decode().strip()
print(out)

time.sleep(8)

# Check process
stdin, stdout, stderr = ssh.exec_command("cat /tmp/pretrain_daily_pid.txt 2>/dev/null && echo '---' && ps aux | grep pretrain_daily | grep -v grep", timeout=10)
print("Process check:")
print(stdout.read().decode().strip())

# Check log
stdin, stdout, stderr = ssh.exec_command("cat /root/project/pretrain_daily_stdout.log 2>/dev/null | head -60", timeout=10)
print("\nLog output:")
print(stdout.read().decode().strip())

ssh.close()
print("\nDone.")
