"""Deploy and run Phase 4 evaluation on AutoDL."""
import paramiko
import os
import time

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

LOCAL_BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = "/root/miniconda3/bin/python"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
sftp = ssh.open_sftp()


def run(cmd, timeout=30):
    print(f">>> {cmd}")
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err and "WARNING" not in err:
        print(f"[stderr] {err[:500]}")
    return out


def upload(local, remote):
    print(f"  Upload: {os.path.basename(local)}")
    sftp.put(local, remote)


# 1. Create evaluation directory
print("=" * 60)
print("Step 1: Setup remote directories")
run("mkdir -p /root/project/model/evaluation")
run("mkdir -p /root/project/outputs/phase4")

# 2. Upload evaluation module files
print("\n" + "=" * 60)
print("Step 2: Upload evaluation modules")
eval_dir = os.path.join(LOCAL_BASE, "model", "evaluation")
for fname in ["__init__.py", "eval_framework.py", "baselines.py", "robustness.py", "backtest.py"]:
    local_path = os.path.join(eval_dir, fname)
    if os.path.exists(local_path):
        upload(local_path, f"/root/project/model/evaluation/{fname}")
    elif fname == "__init__.py":
        sftp.file(f"/root/project/model/evaluation/__init__.py", "w").close()
        print(f"  Created empty __init__.py")

# 3. Upload CoT results
print("\n" + "=" * 60)
print("Step 3: Upload CoT results")
cot_path = os.path.join(LOCAL_BASE, "model", "outputs", "cot", "cot_results.json")
if os.path.exists(cot_path):
    run("mkdir -p /root/project/model/outputs/cot")
    upload(cot_path, "/root/project/model/outputs/cot/cot_results.json")
else:
    print("  CoT results not found locally, checking remote...")
    run("ls -la /root/project/model/outputs/cot/cot_results.json 2>/dev/null || echo NOT_FOUND")

# 4. Verify data files
print("\n" + "=" * 60)
print("Step 4: Verify data")
run("ls /root/project/data/processed/extracted_cases/*.json 2>/dev/null | wc -l")
run("ls /root/project/data/raw/astock/ 2>/dev/null | wc -l")
run("ls /root/project/outputs/finetune/L0/fold_0/best.pt 2>/dev/null")
run("ls /root/project/outputs/finetune/L3/fold_0/best.pt 2>/dev/null")

# 5. Install sklearn if needed
print("\n" + "=" * 60)
print("Step 5: Ensure dependencies")
run(f"{PYTHON} -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || {PYTHON} -m pip install scikit-learn -q", timeout=60)

# 6. Create remote Phase 4 runner
print("\n" + "=" * 60)
print("Step 6: Create Phase 4 runner")

script = r'''#!/usr/bin/env python3
"""Phase 4: Full evaluation on AutoDL with correct paths."""
import sys, json, time, numpy as np, torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/root/project/outputs/phase4/phase4.log", level="DEBUG")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from torch.utils.data import DataLoader

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
OUTPUT_DIR = Path("/root/project/outputs/phase4")
FINETUNE_DIR = Path("/root/project/outputs/finetune")
COT_PATH = Path("/root/project/model/outputs/cot/cot_results.json")

MODEL_CFG = {
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001},
}
FEAT_CFG = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}

torch.manual_seed(42)
np.random.seed(42)

def custom_collate(batch):
    elem = batch[0]
    result = {}
    for key in elem:
        vals = [d[key] for d in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result

def load_cases():
    cases = []
    for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
        with open(f, "r", encoding="utf-8") as fh:
            case = json.load(fh)
            if case.get("stock_code") and case.get("manipulation_start_date"):
                cases.append(case)
    return cases

def build_datasets(train_cases, val_cases, cot_emb=None):
    ds_args = dict(
        astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
        cot_embeddings=cot_emb or {},
    )
    return AStockFinetuneDataset(train_cases, **ds_args), AStockFinetuneDataset(val_cases, **ds_args)

def predict_with_model(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch_gpu)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_gpu["label"].cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def load_moe_model(level, level_cfg, n_features, fold=0, device="cpu"):
    ckpt_path = FINETUNE_DIR / level / f"fold_{fold}" / "best.pt"
    model = FineTuneModel.from_pretrained(
        "", MODEL_CFG, n_features=n_features,
        level=level, level_cfg=level_cfg,
        cot_cfg={"embedding_dim": 384, "projection_dim": 128},
    )
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model


# ========== SECTION 1: Comparison ==========
def run_comparison(cases, device):
    from evaluation.eval_framework import full_evaluation
    from evaluation.baselines import run_traditional_ml, LSTMClassifier, TransformerClassifier, train_dl_baseline

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 1: Comparison Experiments")

    splits = build_cv_splits(cases, n_folds=5, seed=42)
    all_results = defaultdict(list)

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        logger.info(f"\n--- CV Fold {fold_idx+1}/5 ---")
        train_ds, val_ds = build_datasets(train_cases, val_cases)
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        n_features = train_ds.get_n_features()
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=custom_collate)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate)

        # Traditional ML
        ml_results = run_traditional_ml(train_ds, val_ds)
        for name, res in ml_results.items():
            metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name=name)
            all_results[name].append(metrics)

        # LSTM
        lstm = LSTMClassifier(n_features=n_features, d_hidden=128, n_layers=2)
        res = train_dl_baseline(lstm, train_loader, val_loader, device, name="LSTM", epochs=50)
        metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="LSTM")
        all_results["LSTM"].append(metrics)

        # Transformer (no MoE)
        transformer = TransformerClassifier(n_features=n_features, d_model=256, n_heads=8, n_layers=6)
        res = train_dl_baseline(transformer, train_loader, val_loader, device, name="Transformer", epochs=50)
        metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="Transformer")
        all_results["Transformer"].append(metrics)

        # MoE No Pretrain (from scratch)
        moe_scratch = FineTuneModel.from_pretrained(
            "", MODEL_CFG, n_features=n_features,
            level="L0", level_cfg={},
            cot_cfg={"embedding_dim": 384, "projection_dim": 128},
        )
        res = train_dl_baseline(moe_scratch, train_loader, val_loader, device, name="MoE-NoPretrain", epochs=50)
        metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="MoE-NoPretrain")
        all_results["MoE-NoPretrain"].append(metrics)

        # Trained MoE L0-L3
        for level, level_cfg in [
            ("L0", {}),
            ("L1", {}),
            ("L2", {"custom_router": True, "metadata_dim": 8}),
            ("L3", {"custom_router": True, "metadata_dim": 8, "expert_freeze": True, "freeze_top_n_experts": 4}),
        ]:
            model = load_moe_model(level, level_cfg, n_features, fold=fold_idx, device=device)
            y_true, y_pred, y_prob = predict_with_model(model, val_loader, device)
            metrics = full_evaluation(y_true, y_pred, y_prob, name=f"MoE-{level}")
            all_results[f"MoE-{level}"].append(metrics)

    avg_results = {}
    for name, fold_metrics in all_results.items():
        numeric_keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
        avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in numeric_keys}
        std = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in numeric_keys}
        avg_results[name] = {**avg, **std}
    return avg_results


# ========== SECTION 2: Robustness ==========
def run_robustness(cases, device):
    from evaluation.eval_framework import full_evaluation
    from evaluation.robustness import split_by_time, split_by_market_cap, split_by_manipulation_type, split_by_board, summarize_case_metadata

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 2: Robustness Tests")

    metadata = summarize_case_metadata(cases, ASTOCK_DIR)
    results = {}

    for split_name, groups in [
        ("time_window", split_by_time(cases)),
        ("market_cap", split_by_market_cap(cases, ASTOCK_DIR)),
        ("manipulation_type", split_by_manipulation_type(cases)),
        ("board_type", split_by_board(cases)),
    ]:
        logger.info(f"\n--- {split_name} ---")
        group_results = {}
        for group_name, group_cases in groups.items():
            if len(group_cases) < 3:
                logger.warning(f"  {group_name}: only {len(group_cases)} cases, skipping")
                continue
            logger.info(f"  {group_name}: {len(group_cases)} cases")
            ds = AStockFinetuneDataset(
                group_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
                min_bars=30, overlap_ratio=0.5, oversample_positive=1, cot_embeddings={},
            )
            if len(ds) == 0:
                continue
            n_features = ds.get_n_features()
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate)
            model = load_moe_model("L3", {"custom_router": True, "metadata_dim": 8, "expert_freeze": True, "freeze_top_n_experts": 4}, n_features, fold=0, device=device)
            y_true, y_pred, y_prob = predict_with_model(model, loader, device)
            metrics = full_evaluation(y_true, y_pred, y_prob, name=f"{split_name}/{group_name}")
            group_results[group_name] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            group_results[group_name]["n_cases"] = len(group_cases)
            group_results[group_name]["n_windows"] = len(ds)
        results[split_name] = group_results
    return {"metadata": metadata, "results": results}


# ========== SECTION 3: Backtesting ==========
def run_backtesting(cases, device):
    from evaluation.backtest import run_backtest

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 3: Backtesting")

    full_ds = AStockFinetuneDataset(
        cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1, cot_embeddings={},
    )
    if len(full_ds) == 0:
        return {}

    n_features = full_ds.get_n_features()
    model = load_moe_model("L3", {"custom_router": True, "metadata_dim": 8, "expert_freeze": True, "freeze_top_n_experts": 4}, n_features, fold=0, device=device)

    predictions_by_case = defaultdict(list)
    model.eval()
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

    logger.info(f"Predictions for {len(predictions_by_case)} cases")
    return run_backtest(cases, dict(predictions_by_case), ASTOCK_DIR)


# ========== SECTION 4: Interpretability ==========
def run_interpretability(cases):
    from evaluation.eval_framework import evaluate_interpretability

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 4: Interpretability")

    if not COT_PATH.exists():
        logger.warning("No CoT results found")
        return {}
    with open(COT_PATH, "r", encoding="utf-8") as f:
        cot_results = json.load(f)
    return evaluate_interpretability(cot_results, cases)


# ========== MAIN ==========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Phase 4 | Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    logger.info(f"Loaded {len(cases)} cases")

    all_results = {}

    # 1. Comparison
    t0 = time.time()
    comparison = run_comparison(cases, device)
    all_results["comparison"] = comparison
    logger.info(f"Comparison: {time.time()-t0:.0f}s")
    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # 2. Ablation (derived from comparison)
    ablation_map = {
        "Full Model (L3)": "MoE-L3",
        "w/o Crypto Pretrain": "MoE-NoPretrain",
        "w/o CoT Distill (L0)": "MoE-L0",
        "w/o Routing+Freeze (L1)": "MoE-L1",
        "w/o Expert Freeze (L2)": "MoE-L2",
    }
    ablation = {k: comparison.get(v, {}) for k, v in ablation_map.items()}
    all_results["ablation"] = ablation
    with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2, default=str)

    # 3. Robustness
    t0 = time.time()
    robustness = run_robustness(cases, device)
    all_results["robustness"] = robustness
    logger.info(f"Robustness: {time.time()-t0:.0f}s")
    with open(OUTPUT_DIR / "robustness_results.json", "w") as f:
        json.dump(robustness, f, indent=2, default=str)

    # 4. Backtesting
    t0 = time.time()
    backtest = run_backtesting(cases, device)
    all_results["backtest"] = backtest
    logger.info(f"Backtest: {time.time()-t0:.0f}s")
    with open(OUTPUT_DIR / "backtest_results.json", "w") as f:
        json.dump(backtest, f, indent=2, default=str)

    # 5. Interpretability
    interp = run_interpretability(cases)
    all_results["interpretability"] = interp
    with open(OUTPUT_DIR / "interpretability_results.json", "w") as f:
        json.dump(interp, f, indent=2, default=str)

    # Save all
    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4 Complete!")
    logger.info(f"{'Method':<25} {'F1':>8} {'PR-AUC':>8} {'ROC-AUC':>8} {'MCC':>8}")
    logger.info("-" * 60)
    for name, m in sorted(comparison.items(), key=lambda x: x[1].get("f1", 0), reverse=True):
        logger.info(f"{name:<25} {m.get('f1',0):>8.4f} {m.get('pr_auc',0):>8.4f} "
                     f"{m.get('roc_auc',0):>8.4f} {m.get('mcc',0):>8.4f}")
    logger.info("=" * 60)
'''

with sftp.file("/root/project/run_phase4.py", "w") as f:
    f.write(script)
print("Phase 4 runner script created.")

# 7. Launch
print("\n" + "=" * 60)
print("Step 7: Launch Phase 4")
try:
    ssh.exec_command(
        f"cd /root/project && nohup {PYTHON} -u run_phase4.py > phase4_stdout.log 2>&1 &",
        timeout=3,
    )
except Exception:
    pass
time.sleep(5)

# 8. Verify
print("\n" + "=" * 60)
print("Step 8: Verify")
run("ps aux | grep run_phase4 | grep -v grep")
run("head -10 /root/project/phase4_stdout.log 2>/dev/null")

sftp.close()
ssh.close()
print("\nPhase 4 deployed and running on AutoDL!")
