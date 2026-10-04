"""Deploy clean rerun: explicit A-share filtering + full L0-L3 + baselines + Phase4."""
import paramiko
import os
import time

HOST = "connect.westd.seetacloud.com"
PORT = 28784
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]
PYTHON = "/root/miniconda3/bin/python"

LOCAL_BASE = os.path.dirname(os.path.abspath(__file__))

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
        print(out[:500])
    if err and "WARNING" not in err.upper():
        print(f"  [stderr] {err[:300]}")
    return out


def upload(local_path, remote_path):
    remote_dir = "/".join(remote_path.split("/")[:-1])
    run(f"mkdir -p {remote_dir}")
    time.sleep(0.2)
    sftp.put(local_path, remote_path)
    sz = os.path.getsize(local_path) / 1024
    print(f"  Uploaded: {os.path.basename(local_path)} ({sz:.0f} KB)")


# ============================================================
# Step 1: Upload latest code fixes
# ============================================================
print("=" * 60)
print("Step 1: Upload latest code")
print("=" * 60)

code_files = [
    ("model/models/moe.py", "model/models/moe.py"),
    ("model/models/ts_moe_encoder.py", "model/models/ts_moe_encoder.py"),
    ("model/models/finetune_model.py", "model/models/finetune_model.py"),
    ("model/data/astock_dataset.py", "model/data/astock_dataset.py"),
    ("model/evaluation/baselines.py", "model/evaluation/baselines.py"),
    ("model/evaluation/eval_framework.py", "model/evaluation/eval_framework.py"),
    ("model/evaluation/robustness.py", "model/evaluation/robustness.py"),
    ("model/evaluation/backtest.py", "model/evaluation/backtest.py"),
    ("model/analysis/expert_analysis.py", "model/analysis/expert_analysis.py"),
    ("model/configs/finetune_config.yaml", "model/configs/finetune_config.yaml"),
    ("model/training/finetune_trainer.py", "model/training/finetune_trainer.py"),
]

for local_rel, remote_rel in code_files:
    local = os.path.join(LOCAL_BASE, local_rel)
    remote = f"/root/project/{remote_rel}"
    if os.path.exists(local):
        upload(local, remote)
    else:
        print(f"  SKIP (not found): {local_rel}")


# ============================================================
# Step 2: Ensure dependencies
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Dependencies")
print("=" * 60)
run(f"{PYTHON} -m pip install xgboost scikit-learn loguru -q", timeout=120)


# ============================================================
# Step 3: Create the clean rerun script
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Create clean rerun script")
print("=" * 60)

# Valid A-share case doc_ids (23 cases: 17 manipulation + 6 insider trading)
VALID_CASES = [
    "CSRC-2017-064", "CSRC-2017-077", "CSRC-2017-098",
    "CSRC-2018-002", "CSRC-2018-012", "CSRC-2018-014", "CSRC-2018-022",
    "CSRC-2021-124", "CSRC-2022-064",
    "CSRC-2023-005", "CSRC-2023-060", "CSRC-2023-077", "CSRC-2023-083",
    "CSRC-2024-008", "CSRC-2024-023", "CSRC-2024-058", "CSRC-2024-059", "CSRC-2024-082",
    "CSRC-2025-003",
    "CSRC-URL-7532746", "CSRC-URL-7532795", "CSRC-URL-7564434", "CSRC-URL-7610753",
]

valid_cases_str = str(VALID_CASES)

rerun_script = f'''#!/usr/bin/env python3
"""Clean rerun: L0-L3 fine-tuning + all baselines + Phase4 evaluation.

Uses explicit A-share case whitelist (23 cases: 17 manipulation + 6 insider trading).
All experiments use consistent parameters: alpha=0.25, overlap_ratio=0.5.
"""
import sys, json, os, time, torch, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")
os.chdir("/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/root/project/outputs/clean_rerun/rerun.log", rotation="50MB", level="DEBUG")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from torch.utils.data import DataLoader

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
COT_PATH = "/root/project/model/outputs/cot/cot_results.json"
PRETRAIN_CKPT = "/root/project/outputs/pretrain/round3_full/best.pt"
OUTPUT_DIR = Path("/root/project/outputs/clean_rerun")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_CASE_IDS = {valid_cases_str}

MODEL_CFG = {{
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {{"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001}},
}}
FEAT_CFG = {{"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}}

torch.manual_seed(42)
np.random.seed(42)

def custom_collate(batch):
    result = {{}}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result

# Load only whitelisted A-share cases
cases = []
for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
        doc_id = c.get("doc_id", "")
        if doc_id in VALID_CASE_IDS:
            code = c.get("stock_code", "")
            if code and c.get("manipulation_start_date"):
                astock_path = Path(ASTOCK_DIR) / doc_id / "daily.parquet"
                if astock_path.exists():
                    cases.append(c)
                    logger.info(f"  Loaded: {{doc_id}} ({{code}})")
                else:
                    logger.warning(f"  Skip (no astock data): {{doc_id}}")
            else:
                logger.warning(f"  Skip (no code/dates): {{doc_id}}")
        else:
            logger.debug(f"  Excluded: {{doc_id}}")

logger.info(f"Loaded {{len(cases)}} valid A-share cases")

# Load CoT embeddings
cot_embeddings = {{}}
if Path(COT_PATH).exists():
    with open(COT_PATH) as f:
        cot_data = json.load(f)
    for entry in cot_data:
        doc_id = entry.get("doc_id", "")
        emb = entry.get("embedding")
        if doc_id and emb:
            cot_embeddings[doc_id] = np.array(emb, dtype=np.float32)
logger.info(f"CoT embeddings: {{len(cot_embeddings)}}")

splits = build_cv_splits(cases, n_folds=5, seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {{device}}")

from training.finetune_trainer import FocalLoss

LEVELS = {{
    "L0": {{"custom_router": False, "expert_freeze": False}},
    "L1": {{"custom_router": False, "expert_freeze": False}},
    "L2": {{"custom_router": True, "metadata_dim": 8, "expert_freeze": False}},
    "L3": {{"custom_router": True, "metadata_dim": 8, "expert_freeze": True, "freeze_top_n_experts": 4}},
}}
LOSS_WEIGHTS = {{
    "L0": {{"classification": 1.0, "cot_distill": 0.0}},
    "L1": {{"classification": 0.7, "cot_distill": 0.3}},
    "L2": {{"classification": 0.7, "cot_distill": 0.3}},
    "L3": {{"classification": 0.7, "cot_distill": 0.3}},
}}

# =====================================================
# PART 1: L0-L3 Fine-tuning
# =====================================================
logger.info("\\n" + "=" * 60)
logger.info("PART 1: L0-L3 Fine-tuning (23 A-share cases)")
logger.info("=" * 60)

all_results = {{}}

for level_name in ["L0", "L1", "L2", "L3"]:
    level_cfg = LEVELS[level_name]
    loss_w = LOSS_WEIGHTS[level_name]
    level_dir = OUTPUT_DIR / level_name
    level_dir.mkdir(exist_ok=True)

    fold_metrics = []
    logger.info(f"\\n{{'=' * 60}}\\n  {{level_name}}: {{level_cfg}}\\n{{'=' * 60}}")

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        fold_dir = level_dir / f"fold_{{fold_idx}}"
        fold_dir.mkdir(exist_ok=True)

        train_ds = AStockFinetuneDataset(
            train_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1,
            cot_embeddings=cot_embeddings if loss_w["cot_distill"] > 0 else {{}},
        )
        val_ds = AStockFinetuneDataset(
            val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1,
            cot_embeddings=cot_embeddings if loss_w["cot_distill"] > 0 else {{}},
        )

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)

        n_features = train_ds.get_n_features()
        model = FineTuneModel.from_pretrained(
            PRETRAIN_CKPT, MODEL_CFG, n_features=n_features,
            level=level_name, level_cfg=level_cfg,
            cot_cfg={{"embedding_dim": 384, "projection_dim": 128}},
        )

        if level_name == "L3" and fold_idx == 0:
            model.to(device)
            act_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)
            act_info = model.get_expert_activations(act_loader, device)
            expert_ranks = act_info["expert_ranks"]
            logger.info(f"Expert ranks: {{expert_ranks}}")
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
                batch_gpu = {{k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}}
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

            model.eval()
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch_gpu = {{k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}}
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
                best_state = {{k: v.cpu().clone() for k, v in model.state_dict().items()}}
                torch.save({{"model_state_dict": best_state, "epoch": epoch, "f1": f1}},
                           fold_dir / "best.pt")
            else:
                patience += 1
                if patience >= 15:
                    break

            if epoch % 10 == 0:
                logger.info(f"  {{level_name}} fold{{fold_idx}} ep{{epoch}}: loss={{total_loss/n:.4f}} f1={{f1:.4f}}")

        if best_state:
            model.load_state_dict(best_state)
        model.to(device).eval()

        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = {{k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}}
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
        metrics = {{
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }}
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = 0.5

        fold_metrics.append(metrics)
        logger.info(f"  {{level_name}} fold{{fold_idx}}: F1={{metrics['f1']:.4f}} MCC={{metrics['mcc']:.4f}} AUC={{metrics['roc_auc']:.4f}}")

        with open(fold_dir / "history.json", "w") as f:
            json.dump({{"final_metrics": metrics}}, f, indent=2)

    avg = {{}}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        avg[key] = float(np.mean(vals))
        avg[f"{{key}}_std"] = float(np.std(vals))

    all_results[f"MoE-{{level_name}}"] = avg
    logger.info(f"\\n{{level_name}} AVG: F1={{avg['f1']:.4f}}+/-{{avg['f1_std']:.4f}} MCC={{avg['mcc']:.4f}}+/-{{avg['mcc_std']:.4f}}")

    with open(level_dir / "summary.json", "w") as f:
        json.dump(avg, f, indent=2)

# =====================================================
# PART 2: Baselines (Traditional ML + DL + MoE-NoPretrain)
# =====================================================
logger.info("\\n" + "=" * 60)
logger.info("PART 2: Baselines")
logger.info("=" * 60)

from evaluation.baselines import run_traditional_ml, LSTMClassifier, TransformerClassifier, train_dl_baseline
from evaluation.eval_framework import full_evaluation

baseline_results = defaultdict(list)

for fold_idx, (train_cases, val_cases) in enumerate(splits):
    logger.info(f"\\n--- Baseline Fold {{fold_idx+1}}/5 ---")
    train_ds = AStockFinetuneDataset(
        train_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1, cot_embeddings={{}},
    )
    val_ds = AStockFinetuneDataset(
        val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1, cot_embeddings={{}},
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        continue

    n_features = train_ds.get_n_features()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate)

    # Traditional ML
    ml_results = run_traditional_ml(train_ds, val_ds)
    for name, res in ml_results.items():
        metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name=name)
        baseline_results[name].append(metrics)

    # LSTM
    lstm = LSTMClassifier(n_features=n_features, d_hidden=128, n_layers=2)
    res = train_dl_baseline(lstm, train_loader, val_loader, device, name="LSTM", epochs=50)
    metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="LSTM")
    baseline_results["LSTM"].append(metrics)

    # Transformer
    transformer = TransformerClassifier(n_features=n_features, d_model=256, n_heads=8, n_layers=6)
    res = train_dl_baseline(transformer, train_loader, val_loader, device, name="Transformer", epochs=50)
    metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="Transformer")
    baseline_results["Transformer"].append(metrics)

    # MoE NoPretrain
    moe_scratch = FineTuneModel.from_pretrained(
        "", MODEL_CFG, n_features=n_features,
        level="L0", level_cfg={{}},
        cot_cfg={{"embedding_dim": 384, "projection_dim": 128}},
    )
    res = train_dl_baseline(moe_scratch, train_loader, val_loader, device, name="MoE-NoPretrain", epochs=50)
    metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="MoE-NoPretrain")
    baseline_results["MoE-NoPretrain"].append(metrics)

for name, fold_metrics in baseline_results.items():
    numeric_keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    avg = {{k: float(np.mean([m[k] for m in fold_metrics])) for k in numeric_keys}}
    std = {{f"{{k}}_std": float(np.std([m[k] for m in fold_metrics])) for k in numeric_keys}}
    all_results[name] = {{**avg, **std}}

# =====================================================
# PART 3: Phase4 - Robustness + Backtest + Ablation
# =====================================================
logger.info("\\n" + "=" * 60)
logger.info("PART 3: Robustness & Backtest")
logger.info("=" * 60)

from evaluation.robustness import split_by_time, split_by_market_cap, split_by_manipulation_type, split_by_board

def load_best_model(level, level_cfg, n_features, fold=0):
    ckpt_path = OUTPUT_DIR / level / f"fold_{{fold}}" / "best.pt"
    model = FineTuneModel.from_pretrained(
        "", MODEL_CFG, n_features=n_features,
        level=level, level_cfg=level_cfg,
        cot_cfg={{"embedding_dim": 384, "projection_dim": 128}},
    )
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model

def predict_with_model(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {{k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}}
            outputs = model(batch_gpu)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_gpu["label"].cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Robustness
full_ds = AStockFinetuneDataset(
    cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
    min_bars=30, overlap_ratio=0.5, oversample_positive=1, cot_embeddings={{}},
)
n_features = full_ds.get_n_features()
l3_cfg = {{"custom_router": True, "metadata_dim": 8, "expert_freeze": True, "freeze_top_n_experts": 4}}
l3_model = load_best_model("L3", l3_cfg, n_features, fold=0)

robustness_results = {{}}
for split_name, groups in [
    ("time_window", split_by_time(cases)),
    ("market_cap", split_by_market_cap(cases, ASTOCK_DIR)),
    ("manipulation_type", split_by_manipulation_type(cases)),
    ("board_type", split_by_board(cases)),
]:
    logger.info(f"  Robustness: {{split_name}}")
    group_results = {{}}
    for group_name, group_cases in groups.items():
        if len(group_cases) < 3:
            continue
        ds = AStockFinetuneDataset(
            group_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1, cot_embeddings={{}},
        )
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate)
        y_true, y_pred, y_prob = predict_with_model(l3_model, loader)
        metrics = full_evaluation(y_true, y_pred, y_prob, name=f"{{split_name}}/{{group_name}}")
        group_results[group_name] = {{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
        group_results[group_name]["n_cases"] = len(group_cases)
    robustness_results[split_name] = group_results

# Backtest
from evaluation.backtest import run_backtest

predictions_by_case = defaultdict(list)
l3_model.eval()
with torch.no_grad():
    for i in range(len(full_ds)):
        sample = full_ds[i]
        tensor_batch = {{
            k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }}
        outputs = l3_model(tensor_batch)
        prob = torch.softmax(outputs["logits"], dim=-1)[0, 1].item()
        pred = 1 if prob > 0.5 else 0
        doc_id = sample.get("doc_id", "")
        date_str = sample.get("window_start_date", "")
        if doc_id and isinstance(doc_id, str):
            predictions_by_case[doc_id].append((date_str, pred, prob))

backtest_results = run_backtest(cases, dict(predictions_by_case), ASTOCK_DIR)

# =====================================================
# PART 4: Save all results
# =====================================================
logger.info("\\n" + "=" * 60)
logger.info("PART 4: Saving results")
logger.info("=" * 60)

# Ablation
ablation_map = {{
    "Full Model (L3)": "MoE-L3",
    "w/o Crypto Pretrain": "MoE-NoPretrain",
    "w/o CoT Distill (L0)": "MoE-L0",
    "w/o Routing+Freeze (L1)": "MoE-L1",
    "w/o Expert Freeze (L2)": "MoE-L2",
}}
ablation = {{k: all_results.get(v, {{}}) for k, v in ablation_map.items()}}

final = {{
    "comparison": dict(all_results),
    "ablation": ablation,
    "robustness": robustness_results,
    "backtest": backtest_results,
    "dataset_info": {{
        "total_cases": len(cases),
        "manipulation_cases": 17,
        "insider_trading_cases": 6,
        "valid_a_share": True,
    }},
}}

with open(OUTPUT_DIR / "all_results.json", "w") as f:
    json.dump(final, f, indent=2, default=str)
with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
    json.dump(dict(all_results), f, indent=2, default=str)
with open(OUTPUT_DIR / "robustness_results.json", "w") as f:
    json.dump(robustness_results, f, indent=2, default=str)
with open(OUTPUT_DIR / "backtest_results.json", "w") as f:
    json.dump(backtest_results, f, indent=2, default=str)
with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
    json.dump(ablation, f, indent=2, default=str)

logger.info("\\n" + "=" * 60)
logger.info("ALL DONE!")
logger.info(f"Results in: {{OUTPUT_DIR}}")
logger.info(f"{{'Method':<25}} {{'F1':>8}} {{'MCC':>8}} {{'ROC-AUC':>8}}")
logger.info("-" * 55)
for name in sorted(all_results.keys(), key=lambda x: all_results[x].get("mcc", 0), reverse=True):
    m = all_results[name]
    logger.info(f"{{name:<25}} {{m.get('f1',0):>8.4f}} {{m.get('mcc',0):>8.4f}} {{m.get('roc_auc',0):>8.4f}}")
logger.info("=" * 60)

print(json.dumps(dict(all_results), indent=2))
'''

with sftp.file("/root/project/run_clean_rerun.py", "w") as f:
    f.write(rerun_script)
print("Clean rerun script created.")


# ============================================================
# Step 4: Create output directory and launch
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Launch")
print("=" * 60)

run("mkdir -p /root/project/outputs/clean_rerun")
run("pkill -f 'python.*run_' 2>/dev/null; sleep 1; echo 'Killed old processes'")

cmd = f"cd /root/project && nohup {PYTHON} -u run_clean_rerun.py > clean_rerun_stdout.log 2>&1 &"
ssh.exec_command(cmd, timeout=5)
time.sleep(3)

out = run("ps aux | grep run_clean_rerun | grep -v grep")
if "run_clean_rerun" in out:
    print("\nProcess is running!")
else:
    print("\nWARNING: Process may not have started. Check log:")
    run("head -20 /root/project/clean_rerun_stdout.log 2>/dev/null")

sftp.close()
ssh.close()
print("\n" + "=" * 60)
print("Deployment complete! Monitor with:")
print(f"  ssh -p {PORT} root@{HOST} 'tail -f /root/project/clean_rerun_stdout.log'")
print("=" * 60)
