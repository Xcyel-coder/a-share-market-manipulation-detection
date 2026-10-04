import os
# -*- coding: utf-8 -*-
"""
Extract full metrics for ALL models from server outputs.
L0/L1/L2 fold-level best checkpoints have metrics saved.
L3 + baselines are in comprehensive_results.json.
"""
import paramiko, sys, json
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# 1. Get L0/L1/L2 fold-level metrics from saved checkpoints
extract_script = r"""
import json, torch, numpy as np
from pathlib import Path

output = {}
base = Path("/root/project/outputs/comprehensive")

for level in ["MoE_L0", "MoE_L1", "MoE_L2"]:
    level_dir = base / level
    if not level_dir.exists():
        continue
    fold_metrics = []
    for fold_idx in range(5):
        ckpt_path = level_dir / f"fold_{fold_idx}" / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            m = ckpt.get("metrics", {})
            fold_metrics.append({"fold": fold_idx, "metrics": m})
    
    key = level.replace("_", "-")
    if fold_metrics:
        output[key] = {
            "fold_metrics": fold_metrics,
        }
        # Aggregate
        for mn in ["mcc", "f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc", "specificity"]:
            vals = [fm["metrics"].get(mn, 0) for fm in fold_metrics if mn in fm["metrics"]]
            if vals:
                output[key][mn] = float(np.mean(vals))
                output[key][f"{mn}_std"] = float(np.std(vals))

print(json.dumps(output, indent=2, default=str))
"""

stdin, stdout, stderr = ssh.exec_command(
    f'/root/miniconda3/bin/python -c "{extract_script}"', timeout=30)
print("=== L0/L1/L2 FROM CHECKPOINTS ===")
l012 = stdout.read().decode()
print(l012)
err = stderr.read().decode()
if "Error" in err or "Traceback" in err:
    print(f"STDERR: {err}")

# 2. Get L3 + baselines from comprehensive_results.json
stdin, stdout, stderr = ssh.exec_command(
    "cat /root/project/outputs/comprehensive/comprehensive_results.json 2>/dev/null", timeout=10)
comp = stdout.read().decode()

# Parse and print summary
try:
    comp_data = json.loads(comp)
    l012_data = json.loads(l012) if l012.strip() else {}
    
    # Merge L0/L1/L2 checkpoint data into comprehensive
    for key in ["MoE-L0", "MoE-L1", "MoE-L2"]:
        if key in l012_data:
            comp_data[key] = l012_data[key]
    
    print("\n" + "=" * 100)
    print(f"{'方法':<18} {'MCC':>7} {'±':>6} {'F1':>7} {'±':>6} {'Acc':>7} {'±':>6} {'Prec':>7} {'Recall':>7} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 100)
    
    ranked = sorted(comp_data.items(), key=lambda x: x[1].get("mcc", 0), reverse=True)
    for name, res in ranked:
        mcc = res.get("mcc", 0)
        mcc_s = res.get("mcc_std", 0)
        f1 = res.get("f1", 0)
        f1_s = res.get("f1_std", 0)
        acc = res.get("accuracy", 0)
        acc_s = res.get("accuracy_std", 0)
        prec = res.get("precision", 0)
        rec = res.get("recall", 0)
        roc = res.get("roc_auc", 0)
        pr = res.get("pr_auc", 0)
        print(f"{name:<18} {mcc:>7.4f} {mcc_s:>6.4f} {f1:>7.4f} {f1_s:>6.4f} "
              f"{acc:>7.4f} {acc_s:>6.4f} {prec:>7.4f} {rec:>7.4f} {roc:>8.4f} {pr:>8.4f}")
    print("=" * 100)
    
    # Save merged results locally
    with open(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\results_comprehensive.json", "w") as f:
        json.dump(comp_data, f, indent=2, default=str)
    print("\nMerged results saved locally.")
    
except Exception as e:
    print(f"Parse error: {e}")

ssh.close()
