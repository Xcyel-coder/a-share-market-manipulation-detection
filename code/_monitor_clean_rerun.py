"""Monitor the clean rerun progress."""
import os
import paramiko
import sys

HOST = "connect.westd.seetacloud.com"
PORT = 28784
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)

_, stdout, _ = ssh.exec_command(
    "ps aux | grep run_clean_rerun | grep -v grep | head -1; "
    "echo '---STATUS---'; "
    "grep -E '(PART|AVG|ALL DONE|ERROR|Traceback|fold[0-4]:)' /root/project/clean_rerun_stdout.log 2>/dev/null | tail -30; "
    "echo '---TAIL---'; "
    "tail -5 /root/project/clean_rerun_stdout.log 2>/dev/null",
    timeout=15,
)
print(stdout.read().decode())
ssh.close()
