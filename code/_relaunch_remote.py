"""Fix python path and relaunch L1-L3 training on AutoDL."""
import os
import paramiko
import time

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)


def run(cmd, timeout=30):
    print(f">>> {cmd}")
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out.strip():
        print(out.strip())
    if err.strip() and "WARNING" not in err and "nohup" not in err.lower():
        print(f"[stderr] {err.strip()}")
    return out.strip()


# Find correct python
print("Step 1: Find python binary")
run("which python3")
run("python3 --version")

# Launch with python3
print("\nStep 2: Launch with python3")
run("cd /root/project && nohup python3 -u run_l1l3.py > l1l3_stdout.log 2>&1 &", timeout=5)
time.sleep(5)

# Verify
print("\nStep 3: Verify running")
procs = run("ps aux | grep run_l1l3 | grep -v grep")
if not procs:
    print("Process not found! Checking error log...")
    run("cat /root/project/l1l3_stdout.log 2>/dev/null | head -20")
else:
    print("Process is running!")
    run("cat /root/project/l1l3_stdout.log 2>/dev/null | head -15")

# Check GPU
print("\nStep 4: Check GPU")
run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null")

ssh.close()
print("\nDone.")
