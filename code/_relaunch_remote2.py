"""Find correct python path on AutoDL and launch L1-L3."""
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


# Find python - check common locations
print("=== Finding Python ===")
for path in [
    "/root/miniconda3/bin/python",
    "/root/miniconda3/bin/python3",
    "/usr/bin/python3",
    "/usr/bin/python",
    "/opt/conda/bin/python",
]:
    result = run(f"ls -la {path} 2>/dev/null && {path} --version 2>&1")
    if result and "No such file" not in result:
        PYTHON = path
        print(f"  -> Found: {PYTHON}")
        break
else:
    run("find /root -name 'python*' -type f 2>/dev/null | head -10")
    run("find /opt -name 'python*' -type f 2>/dev/null | head -5")
    run("find /usr -name 'python*' -type f 2>/dev/null | head -5")
    print("Searching PATH-like locations...")
    run("ls /root/miniconda3/bin/python* 2>/dev/null")
    run("echo $PATH", timeout=5)
    PYTHON = None

if not PYTHON:
    print("ERROR: Could not find python!")
    ssh.close()
    exit(1)

# Check torch is available
print(f"\n=== Verify torch with {PYTHON} ===")
run(f"{PYTHON} -c 'import torch; print(f\"torch={{torch.__version__}}, cuda={{torch.cuda.is_available()}}\")'")

# Launch
print(f"\n=== Launching with {PYTHON} ===")
run(f"cd /root/project && nohup {PYTHON} -u run_l1l3.py > l1l3_stdout.log 2>&1 &", timeout=5)
time.sleep(5)

# Verify
print("\n=== Verify ===")
run("ps aux | grep run_l1l3 | grep -v grep")
run("head -15 /root/project/l1l3_stdout.log 2>/dev/null")
run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null")

ssh.close()
print("\nDone.")
