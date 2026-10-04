"""
Deploy CV-based robustness & backtest: collect val-set predictions from each fold,
then split by subgroups. Uses L2 (best model). No data leakage.
"""
import paramiko
import os
import time

HOST = "connect.westd.seetacloud.com"
PORT = 28784
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]
PYTHON = "/root/miniconda3/bin/python"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
sftp = ssh.open_sftp()


def run(cmd, timeout=30):
    print(f">>> {cmd[:120]}")
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out[:600])
    if err and "WARNING" not in err.upper():
        print(f"  [stderr] {err[:300]}")
    return out


# Valid A-share case IDs (23 cases)
VALID_CASES = [
    "CSRC-2017-064", "CSRC-2017-077", "CSRC-2017-098",
    "CSRC-2018-002", "CSRC-2018-012", "CSRC-2018-014", "CSRC-2018-022",
    "CSRC-2021-124", "CSRC-2022-064",
    "CSRC-2023-005", "CSRC-2023-060", "CSRC-2023-077", "CSRC-2023-083",
    "CSRC-2024-008", "CSRC-2024-023", "CSRC-2024-058", "CSRC-2024-059", "CSRC-2024-082",
    "CSRC-2025-003",
    "CSRC-URL-7532746", "CSRC-URL-7532795", "CSRC-URL-7564434", "CSRC-URL-7610753",
]

script = '''#!/usr/bin/env python3
"""
CV-based robustness & backtest: no data leakage.
For each of 5 folds, load L2 fold checkpoint, predict on val set only.
Aggregate all val-set predictions, then split by subgroups.
"""
import sys, json, os, torch, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")
os.chdir("/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from torch.utils.data import DataLoader
from evaluation.eval_framework import full_evaluation
from evaluation.robustness import split_by_time, split_by_market_cap, split_by_manipulation_type, split_by_board
from evaluation.backtest import run_backtest

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
CKPT_DIR = Path("/root/project/outputs/clean_rerun/L2")
OUTPUT_DIR = Path("/root/project/outputs/clean_rerun")

VALID_IDS = set(''' + repr(VALID_CASES) + ''')

MODEL_CFG = {
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001},
}
FEAT_CFG = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}
L2_CFG = {"custom_router": True, "metadata_dim": 8, "expert_freeze": False}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def custom_collate(batch):
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result

# Load cases
cases = []
for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
        doc_id = c.get("doc_id", "")
        code = c.get("stock_code") or ""
        if code == "None": code = ""
        if doc_id in VALID_IDS and code and c.get("manipulation_start_date"):
            if (Path(ASTOCK_DIR) / doc_id / "daily.parquet").exists():
                cases.append(c)

logger.info(f"Loaded {len(cases)} cases")

# Build same CV splits as training
splits = build_cv_splits(cases, n_folds=5, seed=42)

# =====================================================
# PART 1: Collect val-set predictions from each fold
# =====================================================
all_val_predictions = []  # list of (doc_id, window_start_date, label, pred, prob, fold_idx)

for fold_idx, (train_cases, val_cases) in enumerate(splits):
    ckpt_path = CKPT_DIR / f"fold_{fold_idx}" / "best.pt"
    if not ckpt_path.exists():
        logger.warning(f"Skip fold {fold_idx}: no checkpoint")
        continue

    val_ds = AStockFinetuneDataset(
        val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1, cot_embeddings={},
    )
    if len(val_ds) == 0:
        continue

    n_features = val_ds.get_n_features()
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)

    model = FineTuneModel.from_pretrained(
        "", MODEL_CFG, n_features=n_features,
        level="L2", level_cfg=L2_CFG,
        cot_cfg={"embedding_dim": 384, "projection_dim": 128},
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch_gpu)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            preds = (probs > 0.5).long()

            labels = batch_gpu["label"].cpu().numpy()
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            doc_ids = batch.get("doc_id", [""] * len(labels))
            dates = batch.get("window_start_date", [""] * len(labels))

            for i in range(len(labels)):
                all_val_predictions.append({
                    "doc_id": doc_ids[i] if isinstance(doc_ids[i], str) else str(doc_ids[i]),
                    "window_start_date": dates[i] if isinstance(dates[i], str) else str(dates[i]),
                    "label": int(labels[i]),
                    "pred": int(preds_np[i]),
                    "prob": float(probs_np[i]),
                    "fold_idx": fold_idx,
                })

    logger.info(f"Fold {fold_idx}: collected {sum(1 for p in all_val_predictions if p['fold_idx']==fold_idx)} val predictions")
    del model
    torch.cuda.empty_cache()

logger.info(f"Total val predictions: {len(all_val_predictions)}")

# =====================================================
# PART 2: Robustness - split by subgroups
# =====================================================
# Map doc_id -> case info
case_map = {c["doc_id"]: c for c in cases}

# Build subgroup membership for each case
time_groups = split_by_time(cases)
cap_groups = split_by_market_cap(cases, ASTOCK_DIR)
type_groups = split_by_manipulation_type(cases)
board_groups = split_by_board(cases)

# Improved manipulation type mapping that covers all cases
def improved_split_by_type(cases):
    groups = defaultdict(list)
    insider_kw = ["内幕", "未公开信息"]
    manip_kw = ["连续交易", "对倒", "虚假申报", "拉抬", "打压", "洗售", "约定交易", "抢帽子", "尾市"]
    for c in cases:
        methods = c.get("manipulation_methods", [])
        method_str = " ".join(methods)
        desc = c.get("manipulation_description", "")
        full = method_str + " " + desc

        if any(k in full for k in ["连续交易", "对倒", "对敲"]):
            groups["连续交易/对倒"].append(c)
        elif any(k in full for k in insider_kw):
            groups["内幕交易"].append(c)
        elif any(k in full for k in ["虚假申报", "拉抬", "抢帽子"]):
            groups["其他操纵"].append(c)
        else:
            groups["其他"].append(c)
    return dict(groups)

type_groups = improved_split_by_type(cases)

# For each grouping, identify which doc_ids belong to which group
def doc_ids_in_groups(groups):
    result = {}
    for gname, gcases in groups.items():
        result[gname] = set(c["doc_id"] for c in gcases)
    return result

time_doc_ids = doc_ids_in_groups(time_groups)
board_doc_ids = doc_ids_in_groups(board_groups)
type_doc_ids = doc_ids_in_groups(type_groups)
cap_doc_ids = doc_ids_in_groups(cap_groups)

def compute_group_metrics(predictions, group_doc_ids):
    group_results = {}
    for gname, gids in group_doc_ids.items():
        gp = [p for p in predictions if p["doc_id"] in gids]
        if len(gp) < 5:
            continue
        y_true = np.array([p["label"] for p in gp])
        y_pred = np.array([p["pred"] for p in gp])
        y_prob = np.array([p["prob"] for p in gp])
        metrics = full_evaluation(y_true, y_pred, y_prob, name=gname)
        num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        num_metrics["n_cases"] = len(gids)
        num_metrics["n_windows"] = len(gp)
        group_results[gname] = num_metrics
    return group_results

robustness = {
    "time_window": compute_group_metrics(all_val_predictions, time_doc_ids),
    "board_type": compute_group_metrics(all_val_predictions, board_doc_ids),
    "manipulation_type": compute_group_metrics(all_val_predictions, type_doc_ids),
    "market_cap": compute_group_metrics(all_val_predictions, cap_doc_ids),
}

# =====================================================
# PART 3: Backtest using val-set predictions only
# =====================================================
preds_by_case = defaultdict(list)
for p in all_val_predictions:
    preds_by_case[p["doc_id"]].append((p["window_start_date"], p["pred"], p["prob"]))

backtest = run_backtest(cases, dict(preds_by_case), ASTOCK_DIR)

# =====================================================
# PART 4: Save
# =====================================================
with open(OUTPUT_DIR / "robustness_cv.json", "w") as f:
    json.dump(robustness, f, indent=2, default=str, ensure_ascii=False)
with open(OUTPUT_DIR / "backtest_cv.json", "w") as f:
    json.dump(backtest, f, indent=2, default=str, ensure_ascii=False)

# Also log type grouping for verification
logger.info("\\nManipulation type groups:")
for gname, gcases in type_groups.items():
    logger.info(f"  {gname}: {len(gcases)} cases - {[c['doc_id'] for c in gcases]}")

logger.info("\\n=== Robustness (CV-based, L2, val-set only) ===")
for dim, groups in robustness.items():
    for gname, metrics in groups.items():
        logger.info(f"  {dim}/{gname}: MCC={metrics.get('mcc',0):.4f} F1={metrics.get('f1',0):.4f} n={metrics.get('n_windows',0)}")

logger.info(f"\\n=== Backtest (CV-based, L2, val-set only) ===")
logger.info(f"  Warned: {backtest['cases_warned']}/{backtest['total_cases']}")
logger.info(f"  Avg lead days: {backtest['avg_lead_days']:.0f}")
logger.info(f"  Avg avoided loss: {backtest['avg_avoided_loss_pct']:.1f}%")

print(json.dumps({"robustness": robustness, "backtest_summary": {k:v for k,v in backtest.items() if k != "details"}}, indent=2, default=str))
'''

with sftp.file("/root/project/run_cv_robustness.py", "w") as f:
    f.write(script)
print("Script uploaded.")

# Launch
run("pkill -f 'run_cv_robustness' 2>/dev/null; sleep 1; echo done")
cmd = f"cd /root/project && nohup {PYTHON} -u run_cv_robustness.py > cv_robustness_stdout.log 2>&1 &"
ssh.exec_command(cmd, timeout=5)
time.sleep(5)

out = run("ps aux | grep run_cv_robustness | grep -v grep")
if "run_cv_robustness" in out:
    print("\nProcess running!")
else:
    print("\nWARNING: not started")
    run("head -20 /root/project/cv_robustness_stdout.log")

sftp.close()
ssh.close()
print("Deployed!")
