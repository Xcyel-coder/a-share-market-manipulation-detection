"""Direct fine-tuning execution (no subprocess wrapper)."""
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "model"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(str(Path(__file__).parent / "model" / "outputs" / "finetune" / "finetune.log"), level="DEBUG")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from training.finetune_trainer import FineTuneTrainer
from torch.utils.data import DataLoader

BASE = Path(__file__).parent
CASES_DIR = str(BASE / "data_pipeline" / "data" / "processed" / "extracted_cases")
ASTOCK_DIR = str(BASE / "data_pipeline" / "data" / "raw" / "astock")
OUTPUT_DIR = str(BASE / "model" / "outputs" / "finetune")
COT_PATH = str(BASE / "model" / "outputs" / "cot" / "cot_results.json")

MODEL_CFG = {
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001},
}
FEAT_CFG = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}
TRAIN_CFG = {
    "epochs": 50, "lr": 1e-4, "batch_size": 32, "warmup_ratio": 0.1,
    "weight_decay": 0.01, "max_grad_norm": 1.0, "seed": 42,
    "focal_loss": {"gamma": 2.0, "alpha": 0.25},
    "early_stopping": {"patience": 10, "min_delta": 0.001},
}

LEVELS = {
    "L0": {"name": "baseline", "description": "Pre-trained MoE + classification only",
            "loss_weights": {"classification": 1.0, "cot_distill": 0.0},
            "freeze_encoder": False, "custom_router": False, "expert_freeze": False},
    "L1": {"name": "cot_distill", "description": "Classification + CoT distillation",
            "loss_weights": {"classification": 0.7, "cot_distill": 0.3},
            "freeze_encoder": False, "custom_router": False, "expert_freeze": False},
    "L2": {"name": "custom_routing", "description": "Metadata-aware routing + CoT",
            "loss_weights": {"classification": 0.7, "cot_distill": 0.3},
            "freeze_encoder": True, "custom_router": True, "expert_freeze": False, "metadata_dim": 8},
    "L3": {"name": "expert_freeze", "description": "Custom routing + expert freezing",
            "loss_weights": {"classification": 0.7, "cot_distill": 0.3},
            "freeze_encoder": False, "custom_router": True, "expert_freeze": True,
            "freeze_top_n_experts": 4, "metadata_dim": 8},
}

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


def run_level(level, cases, pretrained_path="", cot_embeddings=None):
    level_cfg = LEVELS[level]
    logger.info(f"\n{'='*60}")
    logger.info(f"Level {level}: {level_cfg['description']}")
    logger.info(f"{'='*60}")

    splits = build_cv_splits(cases, n_folds=5, seed=42)
    all_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx+1}/5 (train={len(train_cases)}, val={len(val_cases)}) ---")

        cot_emb = cot_embeddings if level_cfg["loss_weights"].get("cot_distill", 0) > 0 else {}

        train_ds = AStockFinetuneDataset(
            train_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1,
            cot_embeddings=cot_emb or {},
        )
        val_ds = AStockFinetuneDataset(
            val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1,
            cot_embeddings=cot_emb or {},
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning(f"Fold {fold_idx}: empty dataset, skipping")
            continue

        n_features = train_ds.get_n_features()
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

        model = FineTuneModel.from_pretrained(
            pretrained_path, MODEL_CFG, n_features=n_features,
            level=level, level_cfg=level_cfg,
            cot_cfg={"embedding_dim": 384, "projection_dim": 128},
        )

        if level == "L3" and level_cfg.get("expert_freeze"):
            acts = model.get_expert_activations(train_loader, torch.device("cpu"))
            model.freeze_experts(acts["expert_ranks"])

        if level_cfg.get("freeze_encoder"):
            for name, param in model.encoder.named_parameters():
                if "router" not in name:
                    param.requires_grad = False

        cfg = {"training": TRAIN_CFG, "model": MODEL_CFG, "output": {"dir": OUTPUT_DIR}}
        trainer = FineTuneTrainer(cfg, OUTPUT_DIR)
        history = trainer.train(
            model, train_loader, val_loader,
            level=level,
            loss_weights=level_cfg["loss_weights"],
            fold=fold_idx,
        )
        if history.get("best_metrics"):
            all_metrics.append(history["best_metrics"])

    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics if isinstance(m.get(k), (int, float))])
               for k in all_metrics[0] if isinstance(all_metrics[0].get(k), (int, float))}
        logger.info(f"\n{level} Summary: F1={avg.get('f1',0):.4f} AUC={avg.get('auc',0):.4f} "
                     f"P={avg.get('precision',0):.4f} R={avg.get('recall',0):.4f}")
        return avg
    return {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "L3", "all"])
    parser.add_argument("--pretrained", default="", help="Path to pretrained checkpoint")
    args = parser.parse_args()

    cases = load_cases()
    logger.info(f"Loaded {len(cases)} valid cases")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cot_cache = Path(__file__).parent / "model" / "outputs" / "cot" / "cot_embeddings.npz"
    cot_embeddings = {}
    if cot_cache.exists():
        data = np.load(cot_cache)
        cot_embeddings = {k: data[k] for k in data.files}
        logger.info(f"Loaded CoT embeddings for {len(cot_embeddings)} cases")
    else:
        logger.warning("No CoT embeddings cache found!")

    levels = ["L0", "L1", "L2", "L3"] if args.level == "all" else [args.level]
    results = {}

    for level in levels:
        t0 = time.time()
        metrics = run_level(level, cases, pretrained_path=args.pretrained,
                            cot_embeddings=cot_embeddings)
        elapsed = time.time() - t0
        results[level] = metrics
        logger.info(f"{level} completed in {elapsed:.1f}s")

    logger.info(f"\n{'='*60}")
    logger.info("Final Comparison:")
    for lv, m in results.items():
        logger.info(f"  {lv}: F1={m.get('f1',0):.4f} AUC={m.get('auc',0):.4f}")
    logger.info(f"{'='*60}")

    with open(Path(OUTPUT_DIR) / "comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
