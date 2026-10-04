"""Download clean rerun results from new AutoDL instance."""
import paramiko
import json
import os

HOST = "connect.westd.seetacloud.com"
PORT = 28784
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

LOCAL_OUT = os.path.join(os.path.dirname(__file__), "model", "outputs", "clean_rerun")
os.makedirs(LOCAL_OUT, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
sftp = ssh.open_sftp()

files = [
    "all_results.json",
    "comparison_results.json",
    "robustness_results.json",
    "backtest_results.json",
    "ablation_results.json",
    "rerun.log",
]

for fname in files:
    remote = f"/root/project/outputs/clean_rerun/{fname}"
    local = os.path.join(LOCAL_OUT, fname)
    try:
        sftp.get(remote, local)
        sz = os.path.getsize(local)
        print(f"  Downloaded: {fname} ({sz/1024:.1f} KB)")
    except Exception as e:
        print(f"  SKIP: {fname} ({e})")

# Also download L0-L3 summaries
for level in ["L0", "L1", "L2", "L3"]:
    remote = f"/root/project/outputs/clean_rerun/{level}/summary.json"
    local_dir = os.path.join(LOCAL_OUT, level)
    os.makedirs(local_dir, exist_ok=True)
    local = os.path.join(local_dir, "summary.json")
    try:
        sftp.get(remote, local)
        print(f"  Downloaded: {level}/summary.json")
    except Exception as e:
        print(f"  SKIP: {level}/summary.json ({e})")

sftp.close()

# Print comparison results
comp_path = os.path.join(LOCAL_OUT, "comparison_results.json")
if os.path.exists(comp_path):
    with open(comp_path, encoding="utf-8") as f:
        data = json.load(f)
    
    print("\n" + "=" * 70)
    print(f"{'Method':<25} {'F1':>8} {'MCC':>8} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("=" * 70)
    for name in sorted(data.keys(), key=lambda x: data[x].get("mcc", 0), reverse=True):
        m = data[name]
        print(f"{name:<25} {m.get('f1',0):>8.4f} {m.get('mcc',0):>8.4f} "
              f"{m.get('roc_auc',0):>8.4f} {m.get('pr_auc',0):>8.4f}")

# Print backtest summary
bt_path = os.path.join(LOCAL_OUT, "backtest_results.json")
if os.path.exists(bt_path):
    with open(bt_path, encoding="utf-8") as f:
        bt = json.load(f)
    print(f"\nBacktest: {bt.get('cases_warned',0)}/{bt.get('total_cases',0)} warned, "
          f"avg lead days: {bt.get('avg_lead_days',0):.0f}, "
          f"avg avoided loss: {bt.get('avg_avoided_loss_pct',0):.1f}%")

ssh.close()
print("\nDone!")
