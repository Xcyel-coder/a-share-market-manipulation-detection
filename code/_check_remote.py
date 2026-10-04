"""Check AutoDL training status and results."""
import os
import paramiko
import json

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)

# 1. Check processes
_, stdout, _ = ssh.exec_command("ps aux | grep python | grep -v grep | grep -v jupyter | grep -v tensorboard", timeout=5)
procs = stdout.read().decode().strip()
print("=== Running Python Processes ===")
print(procs or "(none)")

# 2. Check GPU
_, stdout, _ = ssh.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null", timeout=5)
print(f"\n=== GPU: {stdout.read().decode().strip()} ===")

# 3. Check key log lines
_, stdout, _ = ssh.exec_command(
    "grep -E 'Summary|completed|Loaded pre-trained|ERROR|Traceback' /root/project/outputs/finetune/finetune_remote.log 2>/dev/null | tail -30",
    timeout=10,
)
print("\n=== Key Log Entries ===")
print(stdout.read().decode())

# 4. Check each fold result
print("=== Fold Results ===")
for level in ["L0", "L1", "L2", "L3"]:
    for fold in range(5):
        path = f"/root/project/outputs/finetune/{level}/fold_{fold}/history.json"
        cmd = f"cat {path} 2>/dev/null"
        _, stdout, _ = ssh.exec_command(cmd, timeout=5)
        data = stdout.read().decode().strip()
        if data:
            try:
                hist = json.loads(data)
                bm = hist.get("best_metrics", {})
                f1 = bm.get("f1", 0)
                auc = bm.get("auc", 0)
                print(f"  {level}/fold_{fold}: F1={f1:.4f} AUC={auc:.4f}")
            except json.JSONDecodeError:
                print(f"  {level}/fold_{fold}: CORRUPT JSON")
        else:
            print(f"  {level}/fold_{fold}: NOT FOUND")

# 5. Check comparison
_, stdout, _ = ssh.exec_command("cat /root/project/outputs/finetune/comparison_pretrained.json 2>/dev/null", timeout=5)
comp = stdout.read().decode().strip()
if comp and comp != "":
    print("\n=== Comparison (Pretrained) ===")
    print(comp[:1000])
else:
    print("\n=== Comparison: NOT FOUND (training incomplete) ===")

# 6. Check stderr/crash log
_, stdout, _ = ssh.exec_command("tail -20 /root/project/finetune_stdout.log 2>/dev/null", timeout=5)
print("\n=== Stdout Log (last 20) ===")
print(stdout.read().decode())

ssh.close()
print("\nDone.")
