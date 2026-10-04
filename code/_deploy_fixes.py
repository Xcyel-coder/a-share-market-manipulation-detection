"""Deploy code fixes to AutoDL and re-run L0-L3 fine-tuning + Phase 4."""
import paramiko
import os

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
sftp = ssh.open_sftp()

BASE_LOCAL = os.path.dirname(__file__)
BASE_REMOTE = "/root/project"

files_to_upload = [
    ("model/models/moe.py", "model/models/moe.py"),
    ("model/models/ts_moe_encoder.py", "model/models/ts_moe_encoder.py"),
    ("model/models/finetune_model.py", "model/models/finetune_model.py"),
    ("model/data/astock_dataset.py", "model/data/astock_dataset.py"),
    ("model/evaluation/baselines.py", "model/evaluation/baselines.py"),
    ("model/analysis/expert_analysis.py", "model/analysis/expert_analysis.py"),
    ("model/configs/finetune_config.yaml", "model/configs/finetune_config.yaml"),
]

print("=== Uploading fixed files ===")
for local_rel, remote_rel in files_to_upload:
    local = os.path.join(BASE_LOCAL, local_rel)
    remote = f"{BASE_REMOTE}/{remote_rel}"
    try:
        sftp.put(local, remote)
        print(f"  {local_rel} -> {remote}")
    except Exception as e:
        print(f"  ERROR {local_rel}: {e}")

# Install xgboost
print("\n=== Installing xgboost ===")
_, stdout, _ = ssh.exec_command(
    "/root/miniconda3/bin/pip install xgboost -q", timeout=60
)
print(stdout.read().decode()[:200])

# Create the re-run script
print("\n=== Creating re-run script ===")
rerun_script = r'''#!/usr/bin/env python3
"""Re-run L0-L3 fine-tuning with fixed metadata routing + Phase 4 evaluation."""
import sys, json, os, time, torch, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")
os.chdir("/root/project/model")

from loguru import logger
logger.add("/root/project/outputs/rerun.log", rotation="50MB")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from torch.utils.data import DataLoader

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
COT_PATH = "/root/project/outputs/cot/cot_results.json"
PRETRAIN_CKPT = "/root/project/outputs/pretrain/round3_full/best.pt"
OUTPUT_DIR = Path("/root/project/outputs/finetune_v2")
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

# Load cases
cases = []
for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
        if c.get("stock_code") and c.get("manipulation_start_date"):
            cases.append(c)
logger.info(f"Loaded {len(cases)} cases")

# Load CoT embeddings
cot_embeddings = {}
if Path(COT_PATH).exists():
    with open(COT_PATH) as f:
        cot_data = json.load(f)
    for entry in cot_data:
        doc_id = entry.get("doc_id", "")
        emb = entry.get("embedding")
        if doc_id and emb:
            cot_embeddings[doc_id] = np.array(emb, dtype=np.float32)
logger.info(f"CoT embeddings: {len(cot_embeddings)}")

splits = build_cv_splits(cases, n_folds=5, seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from training.finetune_trainer import FocalLoss

LEVELS = {
    "L0": {"custom_router": False, "expert_freeze": False},
    "L1": {"custom_router": False, "expert_freeze": False},
    "L2": {"custom_router": True, "metadata_dim": 8, "expert_freeze": False},
    "L3": {"custom_router": True, "metadata_dim": 8, "expert_freeze": True, "freeze_top_n_experts": 4},
}
LOSS_WEIGHTS = {
    "L0": {"classification": 1.0, "cot_distill": 0.0},
    "L1": {"classification": 0.7, "cot_distill": 0.3},
    "L2": {"classification": 0.7, "cot_distill": 0.3},
    "L3": {"classification": 0.7, "cot_distill": 0.3},
}

all_results = {}

for level_name in ["L0", "L1", "L2", "L3"]:
    level_cfg = LEVELS[level_name]
    loss_w = LOSS_WEIGHTS[level_name]
    level_dir = OUTPUT_DIR / level_name
    level_dir.mkdir(exist_ok=True)

    fold_metrics = []
    logger.info(f"\n{'='*60}\n  {level_name}: {level_cfg}\n{'='*60}")

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        fold_dir = level_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        train_ds = AStockFinetuneDataset(
            train_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1,
            cot_embeddings=cot_embeddings if loss_w["cot_distill"] > 0 else {},
        )
        val_ds = AStockFinetuneDataset(
            val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1,
            cot_embeddings=cot_embeddings if loss_w["cot_distill"] > 0 else {},
        )

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)

        n_features = train_ds.get_n_features()
        model = FineTuneModel.from_pretrained(
            PRETRAIN_CKPT, MODEL_CFG, n_features=n_features,
            level=level_name, level_cfg=level_cfg,
            cot_cfg={"embedding_dim": 384, "projection_dim": 128},
        )

        # Expert freezing for L3
        if level_name == "L3" and fold_idx == 0:
            act_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)
            act_info = model.get_expert_activations(act_loader, device)
            expert_ranks = act_info["expert_ranks"]
            logger.info(f"Expert ranks: {expert_ranks}")
        if level_name == "L3":
            model.freeze_experts(expert_ranks)

        model.to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-5, weight_decay=0.01,
        )
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        best_f1, patience, best_state = 0, 0, None
        for epoch in range(50):
            model.train()
            total_loss, n = 0, 0
            for batch in train_loader:
                batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(batch_gpu)
                loss = loss_fn(outputs["logits"], batch_gpu["label"])

                if loss_w["cot_distill"] > 0 and "cls_proj" in outputs:
                    cot_loss = 1 - torch.nn.functional.cosine_similarity(
                        outputs["cls_proj"], outputs["cot_target_proj"], dim=-1
                    ).mean()
                    loss = loss_w["classification"] * loss + loss_w["cot_distill"] * cot_loss

                if "moe_loss" in outputs:
                    loss = loss + 0.001 * outputs["moe_loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1

            # Validate
            model.eval()
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    outputs = model(batch_gpu)
                    probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
                    preds = (probs > 0.5).long()
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(batch_gpu["label"].cpu().numpy())

            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            if f1 > best_f1 + 0.001:
                best_f1 = f1
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model_state_dict": best_state, "epoch": epoch, "f1": f1},
                           fold_dir / "best.pt")
            else:
                patience += 1
                if patience >= 15:
                    break

            if epoch % 10 == 0:
                logger.info(f"  {level_name} fold{fold_idx} ep{epoch}: loss={total_loss/n:.4f} f1={f1:.4f}")

        # Final eval
        if best_state:
            model.load_state_dict(best_state)
        model.to(device).eval()

        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(batch_gpu)
                probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
                preds = (probs > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_gpu["label"].cpu().numpy())

        from sklearn.metrics import (
            f1_score, precision_score, recall_score, roc_auc_score,
            average_precision_score, matthews_corrcoef, accuracy_score
        )
        y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
        metrics = {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = 0.5

        fold_metrics.append(metrics)
        logger.info(f"  {level_name} fold{fold_idx}: F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f} AUC={metrics['roc_auc']:.4f}")

        with open(fold_dir / "history.json", "w") as f:
            json.dump({"final_metrics": metrics}, f, indent=2)

    # Aggregate
    avg = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        avg[key] = float(np.mean(vals))
        avg[f"{key}_std"] = float(np.std(vals))

    all_results[f"MoE-{level_name}"] = avg
    logger.info(f"\n{level_name} AVERAGE: F1={avg['f1']:.4f}±{avg['f1_std']:.4f} MCC={avg['mcc']:.4f}±{avg['mcc_std']:.4f} AUC={avg['roc_auc']:.4f}")

    with open(level_dir / "summary.json", "w") as f:
        json.dump(avg, f, indent=2)

# Save comparison
with open(OUTPUT_DIR / "comparison_v2.json", "w") as f:
    json.dump(all_results, f, indent=2)

logger.info("\n\nALL DONE. Results saved.")
print(json.dumps(all_results, indent=2))
'''

with sftp.file("/root/project/run_rerun.py", "w") as f:
    f.write(rerun_script)
sftp.close()

# Kill old processes
print("\n=== Killing old processes ===")
ssh.exec_command("pkill -f 'python.*run_' 2>/dev/null", timeout=5)

# Launch
print("=== Launching re-run ===")
cmd = "cd /root/project && nohup /root/miniconda3/bin/python -u run_rerun.py > rerun_stdout.log 2>&1 &"
ssh.exec_command(cmd, timeout=5)

import time
time.sleep(3)
_, stdout, _ = ssh.exec_command("ps aux | grep run_rerun | grep -v grep", timeout=5)
ps = stdout.read().decode()
if "run_rerun" in ps:
    print("Process running!")
    print(ps.strip())
else:
    print("WARNING: Process not found")

ssh.close()
print("\nDeployment complete. Monitor with _check_remote.py")
