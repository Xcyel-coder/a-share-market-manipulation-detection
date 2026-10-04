"""Check Phase 4 progress on AutoDL."""
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
print(run("ps aux | grep -E 'phase4|python' | grep -v grep | grep -v jupyter") or "(none)")

print("\n=== GPU ===")
print(run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null"))

print("\n=== Phase 4 stdout (last 30) ===")
print(run("tail -30 /root/project/phase4_stdout.log 2>/dev/null") or "(empty)")

print("\n=== Phase 4 log (last 20) ===")
print(run("tail -20 /root/project/outputs/phase4/phase4.log 2>/dev/null") or "(empty)")

print("\n=== Output files ===")
print(run("ls -la /root/project/outputs/phase4/*.json 2>/dev/null") or "(none)")

ssh.close()
print("\nDone.")
