"""Audit AutoDL data against thesis claims."""
import os
import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.westd.seetacloud.com", port=28784, username="root", password=os.environ["AUTODL_SSH_PASSWORD"], timeout=15)

def run(cmd):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    return stdout.read().decode().strip()

print("=== L2 fold checkpoints ===")
print(run("ls -la /root/project/outputs/clean_rerun/L2/fold_*/best.pt 2>/dev/null"))

print("\n=== Comparison results summary ===")
out = run("cat /root/project/outputs/clean_rerun/comparison_results.json")
if out:
    d = json.loads(out)
    for k, v in d.items():
        print(f"  {k}: F1={v.get('f1',0):.4f} MCC={v.get('mcc',0):.4f} ROC={v.get('roc_auc',0):.4f}")

print("\n=== Robustness CV summary ===")
out = run("cat /root/project/outputs/clean_rerun/robustness_cv.json")
if out:
    d = json.loads(out)
    for dim, groups in d.items():
        for gname, metrics in groups.items():
            print(f"  {dim}/{gname}: MCC={metrics.get('mcc',0):.4f} n_cases={metrics.get('n_cases',0)} n_win={metrics.get('n_windows',0)}")

print("\n=== Backtest CV summary ===")
out = run("cat /root/project/outputs/clean_rerun/backtest_cv.json")
if out:
    d = json.loads(out)
    print(f"  warned={d['cases_warned']}/{d['total_cases']} avg_lead={d['avg_lead_days']:.0f} avg_avoided={d['avg_avoided_loss_pct']:.1f}%")
    warned_details = [x for x in d.get("details", []) if x.get("warned")]
    lead_days = sorted([x["lead_days"] for x in warned_details])
    n = len(lead_days)
    median = lead_days[n // 2] if n else 0
    print(f"  lead_days sorted: {lead_days}")
    print(f"  median lead days: {median}")

print("\n=== Count valid cases (from data) ===")
out = run("ls /root/project/data/processed/extracted_cases/*_extracted.json 2>/dev/null | wc -l")
print(f"  Total extracted cases: {out}")

print("\n=== Clean rerun output files ===")
print(run("ls -la /root/project/outputs/clean_rerun/*.json 2>/dev/null"))

print("\n=== CoT embeddings ===")
out = run("python3 -c \"import numpy as np; d=np.load('/root/project/model/outputs/cot/cot_embeddings.npz'); print(f'Keys: {len(d.files)}, shape: {d[d.files[0]].shape}')\" 2>/dev/null")
print(f"  CoT embeddings: {out}")

print("\n=== Window counts from clean_rerun log ===")
out = run("grep 'AStockDataset' /root/project/clean_rerun_stdout.log 2>/dev/null | tail -5")
print(f"  {out}")

ssh.close()
