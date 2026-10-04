"""Re-run NoPretrain MoE + all baselines (ML + DL) with unified parameters.

Ensures consistency: alpha=0.25, overlap_ratio=0.5, same data splits as L0-L3.
"""
import os
import paramiko, time

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

SCRIPT = r'''#!/usr/bin/env python3
"""NoPretrain MoE + all baselines under unified parameters."""
import sys, json, os, torch, numpy as np
from pathlib import Path

sys.path.insert(0, "/root/project/model")
os.chdir("/root/project/model")

from loguru import logger
logger.add("/root/project/outputs/baselines_v2.log", rotation="50MB")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from evaluation.baselines import (
    run_traditional_ml, LSTMClassifier, TransformerClassifier, train_dl_baseline,
)
from torch.utils.data import DataLoader
from training.finetune_trainer import FocalLoss
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, accuracy_score,
)

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
PRETRAIN_CKPT = "/root/project/outputs/pretrain/round3_full/best.pt"
OUTPUT_DIR = Path("/root/project/outputs/baselines_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CFG = {
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001},
}
FEAT_CFG = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}

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

def calc_metrics(y_true, y_pred, y_prob):
    m = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        m["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        m["roc_auc"] = 0.5
        m["pr_auc"] = 0.5
    return m

# Load cases
cases = []
for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
        if c.get("stock_code") and c.get("manipulation_start_date"):
            cases.append(c)
logger.info(f"Loaded {len(cases)} cases")

splits = build_cv_splits(cases, n_folds=5, seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_results = {}

# =====================================================================
# 1. NoPretrain MoE (same arch, random init, alpha=0.25, overlap=0.5)
# =====================================================================
logger.info("=" * 60 + "\n  NoPretrain MoE\n" + "=" * 60)
fold_metrics = []
for fold_idx, (train_cases, val_cases) in enumerate(splits):
    train_ds = AStockFinetuneDataset(
        train_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
    )
    val_ds = AStockFinetuneDataset(
        val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)

    n_features = train_ds.get_n_features()
    # No pretrained checkpoint -> random init
    model = FineTuneModel.from_pretrained(
        None, MODEL_CFG, n_features=n_features,
        level="L0", level_cfg={"custom_router": False, "expert_freeze": False},
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

    best_f1, patience_cnt, best_state = 0, 0, None
    for epoch in range(50):
        model.train()
        total_loss, n = 0, 0
        for batch in train_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch_gpu)
            loss = loss_fn(outputs["logits"], batch_gpu["label"])
            if "moe_loss" in outputs:
                loss = loss + 0.001 * outputs["moe_loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        model.eval()
        preds, probs, labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                bg = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                out = model(bg)
                p = torch.softmax(out["logits"], dim=-1)[:, 1]
                preds.extend((p > 0.5).long().cpu().numpy())
                probs.extend(p.cpu().numpy())
                labels.extend(bg["label"].cpu().numpy())

        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1 + 0.001:
            best_f1 = f1
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()

    preds, probs, labels = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            bg = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(bg)
            p = torch.softmax(out["logits"], dim=-1)[:, 1]
            preds.extend((p > 0.5).long().cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(bg["label"].cpu().numpy())

    m = calc_metrics(np.array(labels), np.array(preds), np.array(probs))
    fold_metrics.append(m)
    logger.info(f"  NoPretrain fold{fold_idx}: F1={m['f1']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

avg = {}
for key in fold_metrics[0]:
    vals = [fm[key] for fm in fold_metrics]
    avg[key] = float(np.mean(vals))
    avg[f"{key}_std"] = float(np.std(vals))
all_results["MoE-NoPretrain"] = avg
logger.info(f"NoPretrain AVG: F1={avg['f1']:.4f} MCC={avg['mcc']:.4f} AUC={avg['roc_auc']:.4f}")

# =====================================================================
# 2. Traditional ML baselines (RF, XGBoost, SVM)
# =====================================================================
logger.info("=" * 60 + "\n  Traditional ML Baselines\n" + "=" * 60)
ml_fold_metrics = {"RandomForest": [], "XGBoost": [], "SVM": []}

for fold_idx, (train_cases, val_cases) in enumerate(splits):
    train_ds = AStockFinetuneDataset(
        train_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
    )
    val_ds = AStockFinetuneDataset(
        val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
    )

    ml_results = run_traditional_ml(train_ds, val_ds)
    for name, res in ml_results.items():
        m = calc_metrics(res["y_true"], res["y_pred"], res["y_prob"])
        ml_fold_metrics[name].append(m)
        logger.info(f"  {name} fold{fold_idx}: F1={m['f1']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")

for name, folds in ml_fold_metrics.items():
    avg = {}
    for key in folds[0]:
        vals = [fm[key] for fm in folds]
        avg[key] = float(np.mean(vals))
        avg[f"{key}_std"] = float(np.std(vals))
    all_results[name] = avg
    logger.info(f"{name} AVG: F1={avg['f1']:.4f} MCC={avg['mcc']:.4f} AUC={avg['roc_auc']:.4f}")

# =====================================================================
# 3. Deep Learning baselines (LSTM, Transformer)
# =====================================================================
logger.info("=" * 60 + "\n  DL Baselines\n" + "=" * 60)
dl_fold_metrics = {"LSTM": [], "Transformer": []}

for fold_idx, (train_cases, val_cases) in enumerate(splits):
    train_ds = AStockFinetuneDataset(
        train_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
    )
    val_ds = AStockFinetuneDataset(
        val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)
    n_features = train_ds.get_n_features()

    # LSTM
    lstm = LSTMClassifier(n_features, d_hidden=128, n_layers=2, dropout=0.2)
    res_lstm = train_dl_baseline(lstm, train_loader, val_loader, device, name="LSTM",
                                  epochs=50, lr=1e-4, patience=10)
    m_lstm = calc_metrics(res_lstm["y_true"], res_lstm["y_pred"], res_lstm["y_prob"])
    dl_fold_metrics["LSTM"].append(m_lstm)
    logger.info(f"  LSTM fold{fold_idx}: F1={m_lstm['f1']:.4f} MCC={m_lstm['mcc']:.4f}")

    # Transformer
    tfm = TransformerClassifier(n_features, d_model=256, n_heads=8, n_layers=6, dropout=0.1)
    res_tfm = train_dl_baseline(tfm, train_loader, val_loader, device, name="Transformer",
                                 epochs=50, lr=1e-4, patience=10)
    m_tfm = calc_metrics(res_tfm["y_true"], res_tfm["y_pred"], res_tfm["y_prob"])
    dl_fold_metrics["Transformer"].append(m_tfm)
    logger.info(f"  Transformer fold{fold_idx}: F1={m_tfm['f1']:.4f} MCC={m_tfm['mcc']:.4f}")

for name, folds in dl_fold_metrics.items():
    avg = {}
    for key in folds[0]:
        vals = [fm[key] for fm in folds]
        avg[key] = float(np.mean(vals))
        avg[f"{key}_std"] = float(np.std(vals))
    all_results[name] = avg
    logger.info(f"{name} AVG: F1={avg['f1']:.4f} MCC={avg['mcc']:.4f} AUC={avg['roc_auc']:.4f}")

# =====================================================================
# Save
# =====================================================================
with open(OUTPUT_DIR / "baselines_v2.json", "w") as f:
    json.dump(all_results, f, indent=2)

logger.info("\n\nALL BASELINES DONE.")
print(json.dumps(all_results, indent=2))
'''

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)

sftp = ssh.open_sftp()
with sftp.file("/root/project/run_baselines_v2.py", "w") as f:
    f.write(SCRIPT)
sftp.close()

print("Script uploaded.")

ssh.exec_command("pkill -f run_rerun 2>/dev/null; pkill -f run_baselines 2>/dev/null", timeout=5)
time.sleep(2)

cmd = "cd /root/project && nohup /root/miniconda3/bin/python -u run_baselines_v2.py > baselines_v2_stdout.log 2>&1 &"
ssh.exec_command(cmd, timeout=5)
time.sleep(3)

_, out, _ = ssh.exec_command("ps aux | grep run_baselines | grep -v grep", timeout=5)
ps = out.read().decode()
print("Running!" if "run_baselines" in ps else "NOT running")
print(ps.strip())

ssh.close()
print("Done.")
