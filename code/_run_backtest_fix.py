"""Fix and re-run backtesting on AutoDL with corrected doc_id path."""
import os
import paramiko
import time

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]
PYTHON = "/root/miniconda3/bin/python"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)


def run(cmd, timeout=30):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err and "WARNING" not in err:
        print(f"[stderr] {err[:300]}")
    return out


# Fix backtest.py on remote
print("=== Fixing backtest.py on AutoDL ===")
fix_cmd = r"""
sed -i 's|parquet_path = Path(astock_dir) / stock_code / "daily.parquet"|parquet_path = Path(astock_dir) / doc_id / "daily.parquet"|' /root/project/model/evaluation/backtest.py
grep 'parquet_path' /root/project/model/evaluation/backtest.py
"""
run(fix_cmd, timeout=10)

# Create and run backtest-only script
print("\n=== Creating backtest re-run script ===")
script = r'''
import sys, json, numpy as np, torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")

from data.astock_dataset import AStockFinetuneDataset
from models.finetune_model import FineTuneModel
from evaluation.backtest import run_backtest

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
FINETUNE_DIR = Path("/root/project/outputs/finetune")
OUTPUT_DIR = Path("/root/project/outputs/phase4")

MODEL_CFG = {
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001},
}
FEAT_CFG = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}

def load_cases():
    cases = []
    for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
        with open(f, "r", encoding="utf-8") as fh:
            case = json.load(fh)
            if case.get("stock_code") and case.get("manipulation_start_date"):
                cases.append(case)
    return cases

cases = load_cases()
print(f"Loaded {len(cases)} cases")

full_ds = AStockFinetuneDataset(
    cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
    min_bars=30, overlap_ratio=0.5, oversample_positive=1, cot_embeddings={},
)
print(f"Dataset: {len(full_ds)} windows")

n_features = full_ds.get_n_features()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use L3 (best model)
level_cfg = {"custom_router": True, "metadata_dim": 8, "expert_freeze": True, "freeze_top_n_experts": 4}
ckpt_path = FINETUNE_DIR / "L3" / "fold_0" / "best.pt"
model = FineTuneModel.from_pretrained(
    "", MODEL_CFG, n_features=n_features,
    level="L3", level_cfg=level_cfg,
    cot_cfg={"embedding_dim": 384, "projection_dim": 128},
)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.to(device).eval()

predictions_by_case = defaultdict(list)
with torch.no_grad():
    for i in range(len(full_ds)):
        sample = full_ds[i]
        tensor_batch = {
            k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        outputs = model(tensor_batch)
        prob = torch.softmax(outputs["logits"], dim=-1)[0, 1].item()
        pred = 1 if prob > 0.5 else 0
        doc_id = sample.get("doc_id", "")
        date_str = sample.get("window_start_date", "")
        if doc_id and isinstance(doc_id, str):
            predictions_by_case[doc_id].append((date_str, pred, prob))

print(f"Predictions for {len(predictions_by_case)} cases")
for did, preds in list(predictions_by_case.items())[:3]:
    warned = sum(1 for _, p, _ in preds if p == 1)
    print(f"  {did}: {len(preds)} windows, {warned} warned")

result = run_backtest(cases, dict(predictions_by_case), ASTOCK_DIR)
print(f"\nBacktest: {result['cases_warned']}/{result['total_cases']} warned")
print(f"Avg avoided loss: {result['avg_avoided_loss_pct']:.2f}%")
print(f"Avg lead days: {result['avg_lead_days']:.1f}")

with open(OUTPUT_DIR / "backtest_results.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print("Saved to backtest_results.json")
'''

sftp = ssh.open_sftp()
with sftp.file("/root/project/run_backtest_fix.py", "w") as f:
    f.write(script)
sftp.close()

# Run
print("\n=== Running backtest ===")
run(f"{PYTHON} -u /root/project/run_backtest_fix.py", timeout=120)

ssh.close()
print("\nDone.")
