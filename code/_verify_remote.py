"""Verify L1-L3 training is running on AutoDL."""
import os
import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)


def run(cmd, timeout=15):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode().strip()


print("=== Processes ===")
print(run("ps aux | grep -E 'run_l1l3|run_finetune' | grep -v grep") or "(none)")

print("\n=== GPU ===")
print(run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null"))

print("\n=== Log (last 20 lines) ===")
print(run("tail -20 /root/project/l1l3_stdout.log 2>/dev/null") or "(empty)")

print("\n=== Detailed log (last 10 lines) ===")
print(run("tail -10 /root/project/outputs/finetune/finetune_l1l3.log 2>/dev/null") or "(empty)")

ssh.close()
print("\nDone.")
