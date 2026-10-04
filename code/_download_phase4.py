"""Download Phase 4 results from AutoDL."""
import os
import paramiko, os, json

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

LOCAL_BASE = os.path.dirname(os.path.abspath(__file__))
LOCAL_P4 = os.path.join(LOCAL_BASE, "model", "outputs", "phase4")
os.makedirs(LOCAL_P4, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
sftp = ssh.open_sftp()

files = [
    "all_results.json", "comparison_results.json", "ablation_results.json",
    "robustness_results.json", "backtest_results.json", "interpretability_results.json",
    "phase4.log",
]

for fname in files:
    remote = f"/root/project/outputs/phase4/{fname}"
    local = os.path.join(LOCAL_P4, fname)
    try:
        sftp.get(remote, local)
        size = os.path.getsize(local)
        print(f"  {fname} ({size:,} bytes)")
    except FileNotFoundError:
        print(f"  {fname} - NOT FOUND")

# Print key results
print("\n=== Comparison Results ===")
with open(os.path.join(LOCAL_P4, "comparison_results.json")) as f:
    comp = json.load(f)
    for name, m in sorted(comp.items(), key=lambda x: x[1].get("f1", 0), reverse=True):
        print(f"  {name:<25} F1={m.get('f1',0):.4f} ROC-AUC={m.get('roc_auc',0):.4f} MCC={m.get('mcc',0):.4f}")

print("\n=== Backtest ===")
with open(os.path.join(LOCAL_P4, "backtest_results.json")) as f:
    print(json.dumps(json.load(f), indent=2))

print("\n=== Interpretability ===")
with open(os.path.join(LOCAL_P4, "interpretability_results.json")) as f:
    print(json.dumps(json.load(f), indent=2))

sftp.close()
ssh.close()
print("\nDone.")
