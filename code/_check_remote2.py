"""Detailed check of L0/L1 results on AutoDL."""
import os
import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)

# L0 summary from detailed log
_, stdout, _ = ssh.exec_command(
    r"grep -E 'L0 Summary|L0 completed|L1 Summary|L1 completed|L2 |L3 |ERROR|Traceback|killed' /root/project/outputs/finetune/finetune_remote.log 2>/dev/null | head -30",
    timeout=10,
)
print("=== Key summaries from log ===")
print(stdout.read().decode())

# Full L1 file listing
_, stdout, _ = ssh.exec_command("find /root/project/outputs/finetune/L1 -type f 2>/dev/null", timeout=5)
print("=== L1 files ===")
print(stdout.read().decode())

# Check if there's an error at the end of log
_, stdout, _ = ssh.exec_command("tail -5 /root/project/outputs/finetune/finetune_remote.log 2>/dev/null", timeout=5)
print("=== Log tail ===")
print(stdout.read().decode())

# Disk space
_, stdout, _ = ssh.exec_command("df -h /root | tail -1", timeout=5)
print(f"=== Disk: {stdout.read().decode().strip()} ===")

ssh.close()
print("Done.")
