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

cmds = [
    "which python python3 2>/dev/null",
    "ls /usr/bin/python* /usr/local/bin/python* 2>/dev/null",
    "ls /root/miniconda3/bin/python* 2>/dev/null",
    "ls /opt/conda/bin/python* 2>/dev/null",
    "conda info --envs 2>/dev/null || echo 'no conda'",
    "echo $PATH",
    "cat /root/.bashrc | grep -i python || echo 'no python in bashrc'",
    "find / -maxdepth 4 -name 'python3*' -type f 2>/dev/null | head -5",
]

for cmd in cmds:
    _, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f"CMD: {cmd}", flush=True)
    if out:
        print(f"  OUT: {out}", flush=True)
    if err:
        print(f"  ERR: {err}", flush=True)

ssh.close()
