"""
Phase 4: Comprehensive Evaluation & Comparison Experiments
Covers: evaluation framework, baselines, ablation, robustness, backtesting.
"""
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "model"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(str(Path(__file__).parent / "model" / "outputs" / "phase4" / "phase4.log"), level="DEBUG")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from torch.utils.data import DataLoader


def custom_collate(batch):
    """Collate that handles mixed tensor/string fields."""
    elem = batch[0]
    result = {}
    for key in elem:
        vals = [d[key] for d in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result

BASE = Path(__file__).parent
CASES_DIR = str(BASE / "data_pipeline" / "data" / "processed" / "extracted_cases")
ASTOCK_DIR = str(BASE / "data_pipeline" / "data" / "raw" / "astock")
OUTPUT_DIR = BASE / "model" / "outputs" / "phase4"
FINETUNE_DIR = BASE / "model" / "outputs" / "finetune"
COT_PATH = BASE / "model" / "outputs" / "cot" / "cot_results.json"

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
    train_ds = AStockFinetuneDataset(train_cases, **ds_args)
    val_ds = AStockFinetuneDataset(val_cases, **ds_args)
    return train_ds, val_ds


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
    """Load a trained MoE model from Phase 3 checkpoint."""
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


# ============================================================
# SECTION 1: Comparison Experiments
# ============================================================

def run_comparison_experiments(cases, device):
    """Run all comparison baselines + MoE levels."""
    from evaluation.eval_framework import full_evaluation
    from evaluation.baselines import run_traditional_ml, LSTMClassifier, TransformerClassifier, train_dl_baseline

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 1: Comparison Experiments")
    logger.info("=" * 60)

    splits = build_cv_splits(cases, n_folds=5, seed=42)
    all_results = defaultdict(list)

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        logger.info(f"\n--- CV Fold {fold_idx + 1}/5 ---")
        train_ds, val_ds = build_datasets(train_cases, val_cases)
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        n_features = train_ds.get_n_features()
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=custom_collate)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate)

        # --- Traditional ML ---
        ml_results = run_traditional_ml(train_ds, val_ds)
        for name, res in ml_results.items():
            metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name=name)
            all_results[name].append(metrics)

        # --- LSTM ---
        lstm = LSTMClassifier(n_features=n_features, d_hidden=128, n_layers=2)
        res = train_dl_baseline(lstm, train_loader, val_loader, device, name="LSTM", epochs=50)
        metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="LSTM")
        all_results["LSTM"].append(metrics)

        # --- Transformer (no MoE) ---
        transformer = TransformerClassifier(n_features=n_features, d_model=256, n_heads=8, n_layers=6)
        res = train_dl_baseline(transformer, train_loader, val_loader, device, name="Transformer", epochs=50)
        metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="Transformer")
        all_results["Transformer"].append(metrics)

        # --- MoE No Pretrain (from scratch) ---
        moe_scratch = FineTuneModel.from_pretrained(
            "", MODEL_CFG, n_features=n_features,
            level="L0", level_cfg={},
            cot_cfg={"embedding_dim": 384, "projection_dim": 128},
        )
        res = train_dl_baseline(moe_scratch, train_loader, val_loader, device, name="MoE-NoPretrain", epochs=50)
        metrics = full_evaluation(res["y_true"], res["y_pred"], res["y_prob"], name="MoE-NoPretrain")
        all_results["MoE-NoPretrain"].append(metrics)

        # --- Trained MoE Levels (load from Phase 3) ---
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


# ============================================================
# SECTION 2: Ablation Matrix
# ============================================================

def run_ablation_matrix(cases, device):
    """
    Ablation experiments mapping to Phase 3 levels:
    - Full model = L3 (pretrain + CoT + routing + freeze)
    - Ablation1 (no pretrain) = MoE-NoPretrain with L3 config
    - Ablation2 (no CoT) = L0 (classification only)
    - Ablation3 (no routing, no freeze) = L1 (CoT only)
    - Ablation4 (no freeze) = L2 (routing only)
    """
    from evaluation.eval_framework import full_evaluation

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 2: Ablation Matrix")
    logger.info("=" * 60)

    ablation_map = {
        "Full Model (L3)": "MoE-L3",
        "w/o Crypto Pretrain": "MoE-NoPretrain",
        "w/o CoT Distill (L0)": "MoE-L0",
        "w/o Routing+Freeze (L1)": "MoE-L1",
        "w/o Expert Freeze (L2)": "MoE-L2",
    }

    return ablation_map


# ============================================================
# SECTION 3: Robustness Tests
# ============================================================

def run_robustness_tests(cases, device):
    """Run model evaluation on different data subgroups."""
    from evaluation.eval_framework import full_evaluation
    from evaluation.robustness import (
        split_by_time, split_by_market_cap,
        split_by_manipulation_type, split_by_board,
        summarize_case_metadata,
    )

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 3: Robustness Tests")
    logger.info("=" * 60)

    metadata_summary = summarize_case_metadata(cases, ASTOCK_DIR)

    robustness_results = {}

    split_configs = [
        ("time_window", split_by_time(cases)),
        ("market_cap", split_by_market_cap(cases, ASTOCK_DIR)),
        ("manipulation_type", split_by_manipulation_type(cases)),
        ("board_type", split_by_board(cases)),
    ]

    for split_name, groups in split_configs:
        logger.info(f"\n--- Robustness: {split_name} ---")
        group_results = {}

        for group_name, group_cases in groups.items():
            if len(group_cases) < 3:
                logger.warning(f"  {group_name}: only {len(group_cases)} cases, skipping")
                continue

            logger.info(f"  {group_name}: {len(group_cases)} cases")
            ds = AStockFinetuneDataset(
                group_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
                min_bars=30, overlap_ratio=0.5, oversample_positive=1,
                cot_embeddings={},
            )
            if len(ds) == 0:
                continue

            n_features = ds.get_n_features()
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate)

            model = load_moe_model("L0", {}, n_features, fold=0, device=device)
            y_true, y_pred, y_prob = predict_with_model(model, loader, device)
            metrics = full_evaluation(y_true, y_pred, y_prob, name=f"{split_name}/{group_name}")

            group_results[group_name] = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            group_results[group_name]["n_cases"] = len(group_cases)
            group_results[group_name]["n_windows"] = len(ds)

        robustness_results[split_name] = group_results

    return {"metadata": metadata_summary, "results": robustness_results}


# ============================================================
# SECTION 4: Backtesting
# ============================================================

def run_backtesting(cases, device):
    """Run backtest using L0 model predictions (per-sample inference)."""
    from evaluation.backtest import run_backtest

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 4: Backtesting")
    logger.info("=" * 60)

    full_ds = AStockFinetuneDataset(
        cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
        cot_embeddings={},
    )
    if len(full_ds) == 0:
        return {}

    n_features = full_ds.get_n_features()
    model = load_moe_model("L0", {}, n_features, fold=0, device=device)

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

    logger.info(f"Generated predictions for {len(predictions_by_case)} cases")
    return run_backtest(cases, dict(predictions_by_case), ASTOCK_DIR)


# ============================================================
# SECTION 5: Interpretability Evaluation
# ============================================================

def run_interpretability_eval(cases):
    """Evaluate CoT interpretability quality."""
    from evaluation.eval_framework import evaluate_interpretability

    logger.info("\n" + "=" * 60)
    logger.info("SECTION 5: Interpretability Evaluation")
    logger.info("=" * 60)

    if not COT_PATH.exists():
        logger.warning("No CoT results found")
        return {}

    with open(COT_PATH, "r", encoding="utf-8") as f:
        cot_results = json.load(f)

    return evaluate_interpretability(cot_results, cases)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Phase 4 Evaluation | Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    logger.info(f"Loaded {len(cases)} cases")

    all_results = {}

    # 1. Comparison experiments
    t0 = time.time()
    comparison = run_comparison_experiments(cases, device)
    all_results["comparison"] = comparison
    logger.info(f"Comparison experiments: {time.time()-t0:.0f}s")

    with open(OUTPUT_DIR / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)

    # 2. Ablation matrix (references comparison results)
    ablation_map = run_ablation_matrix(cases, device)
    ablation_results = {}
    for abl_name, comp_key in ablation_map.items():
        if comp_key in comparison:
            ablation_results[abl_name] = comparison[comp_key]
    all_results["ablation"] = ablation_results

    with open(OUTPUT_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False, default=str)

    # 3. Robustness tests
    t0 = time.time()
    robustness = run_robustness_tests(cases, device)
    all_results["robustness"] = robustness
    logger.info(f"Robustness tests: {time.time()-t0:.0f}s")

    with open(OUTPUT_DIR / "robustness_results.json", "w", encoding="utf-8") as f:
        json.dump(robustness, f, indent=2, ensure_ascii=False, default=str)

    # 4. Backtesting
    t0 = time.time()
    backtest = run_backtesting(cases, device)
    all_results["backtest"] = backtest
    logger.info(f"Backtesting: {time.time()-t0:.0f}s")

    with open(OUTPUT_DIR / "backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(backtest, f, indent=2, ensure_ascii=False, default=str)

    # 5. Interpretability
    interp = run_interpretability_eval(cases)
    all_results["interpretability"] = interp

    with open(OUTPUT_DIR / "interpretability_results.json", "w", encoding="utf-8") as f:
        json.dump(interp, f, indent=2, ensure_ascii=False, default=str)

    # Save combined results
    with open(OUTPUT_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 4 Complete!")
    logger.info("=" * 60)

    # Print comparison table
    logger.info("\nComparison Results:")
    logger.info(f"{'Method':<25} {'F1':>8} {'PR-AUC':>8} {'ROC-AUC':>8} {'MCC':>8}")
    logger.info("-" * 60)
    for name, m in sorted(comparison.items(), key=lambda x: x[1].get("f1", 0)):
        logger.info(f"{name:<25} {m.get('f1',0):>8.4f} {m.get('pr_auc',0):>8.4f} "
                     f"{m.get('roc_auc',0):>8.4f} {m.get('mcc',0):>8.4f}")


if __name__ == "__main__":
    main()
