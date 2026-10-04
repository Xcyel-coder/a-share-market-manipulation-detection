"""
Phase 3 fine-tuning entry point.

Usage:
    python -m scripts.run_finetune --config configs/finetune_config.yaml --level L0
    python -m scripts.run_finetune --config configs/finetune_config.yaml --level L1
    python -m scripts.run_finetune --config configs/finetune_config.yaml --level L2
    python -m scripts.run_finetune --config configs/finetune_config.yaml --level L3
    python -m scripts.run_finetune --config configs/finetune_config.yaml --level all
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from training.finetune_trainer import FineTuneTrainer


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cases(cases_dir: str) -> list[dict]:
    """Load all extracted A-share case JSONs."""
    cases_path = Path(cases_dir)
    cases = []
    for f in sorted(cases_path.glob("*_extracted.json")):
        with open(f, "r", encoding="utf-8") as fh:
            case = json.load(fh)
            if case.get("stock_code") and case.get("manipulation_start_date"):
                cases.append(case)
    logger.info(f"Loaded {len(cases)} valid cases from {cases_dir}")
    return cases


def load_cot_embeddings(cot_path: str, model_name: str) -> dict[str, np.ndarray]:
    """Pre-compute or load CoT embeddings."""
    cache_path = Path(cot_path).parent / "cot_embeddings.npz"

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        embeddings = {k: data[k] for k in data.files}
        logger.info(f"Loaded {len(embeddings)} cached CoT embeddings")
        return embeddings

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping CoT embeddings")
        return {}

    with open(cot_path, "r", encoding="utf-8") as f:
        cot_results = json.load(f)

    st_model = SentenceTransformer(model_name)
    embeddings = {}

    case_cots: dict[str, list[str]] = {}
    for entry in cot_results:
        doc_id = entry["doc_id"]
        text = entry.get("cot_text", "")
        if text:
            case_cots.setdefault(doc_id, []).append(text)

    for doc_id, texts in case_cots.items():
        combined = "\n\n".join(texts[:4])[:8000]
        emb = st_model.encode(combined, normalize_embeddings=True)
        embeddings[doc_id] = emb.astype(np.float32)

    np.savez(cache_path, **embeddings)
    logger.info(f"Computed and cached {len(embeddings)} CoT embeddings")
    return embeddings


def run_level(
    level: str,
    cfg: dict,
    cases: list[dict],
    cot_embeddings: dict,
) -> dict:
    """Run all CV folds for a given level."""
    level_cfg = cfg["levels"][level]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    cot_cfg = cfg.get("cot_distill", {})

    logger.info(f"\n{'='*60}")
    logger.info(f"Level {level}: {level_cfg['description']}")
    logger.info(f"{'='*60}")

    splits = build_cv_splits(cases, n_folds=data_cfg["cv_folds"], seed=train_cfg["seed"])
    all_fold_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx+1}/{len(splits)} "
                     f"(train={len(train_cases)}, val={len(val_cases)}) ---")

        use_cot = level_cfg["loss_weights"].get("cot_distill", 0) > 0
        cot_emb = cot_embeddings if use_cot else {}

        train_ds = AStockFinetuneDataset(
            cases=train_cases,
            astock_dir=data_cfg["astock_dir"],
            feature_cfg=data_cfg["features"],
            seq_len=data_cfg["seq_len"],
            stride=data_cfg["stride"],
            min_bars=data_cfg["min_window_bars"],
            overlap_ratio=data_cfg["manipulation_overlap_ratio"],
            oversample_positive=train_cfg.get("oversample_positive", 1),
            cot_embeddings=cot_emb,
        )

        val_ds = AStockFinetuneDataset(
            cases=val_cases,
            astock_dir=data_cfg["astock_dir"],
            feature_cfg=data_cfg["features"],
            seq_len=data_cfg["seq_len"],
            stride=data_cfg["stride"],
            min_bars=data_cfg["min_window_bars"],
            overlap_ratio=data_cfg["manipulation_overlap_ratio"],
            oversample_positive=1,
            cot_embeddings=cot_emb,
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning(f"Fold {fold_idx}: empty dataset, skipping")
            continue

        n_features = train_ds.get_n_features()

        train_loader = DataLoader(
            train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
            num_workers=0, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
            num_workers=0,
        )

        pretrained_path = cfg["pretrained"]["checkpoint"]
        model = FineTuneModel.from_pretrained(
            checkpoint_path=pretrained_path,
            model_cfg=model_cfg,
            n_features=n_features,
            level=level,
            level_cfg=level_cfg,
            cot_cfg=cot_cfg,
        )

        if level == "L3" and level_cfg.get("expert_freeze"):
            activations = model.get_expert_activations(train_loader, torch.device("cpu"))
            model.freeze_experts(activations["expert_ranks"])
            logger.info(f"Frozen experts: {activations['expert_ranks'][:level_cfg['freeze_top_n_experts']]}")

        if level_cfg.get("freeze_encoder"):
            for name, param in model.encoder.named_parameters():
                if "router" not in name:
                    param.requires_grad = False
            logger.info("Encoder frozen (except routers)")

        trainer = FineTuneTrainer(cfg, cfg["output"]["dir"])
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            level=level,
            loss_weights=level_cfg["loss_weights"],
            fold=fold_idx,
        )

        if history.get("best_metrics"):
            all_fold_metrics.append(history["best_metrics"])

    if all_fold_metrics:
        avg_metrics = {}
        for key in all_fold_metrics[0]:
            vals = [m[key] for m in all_fold_metrics if isinstance(m.get(key), (int, float))]
            if vals:
                avg_metrics[f"avg_{key}"] = np.mean(vals)
                avg_metrics[f"std_{key}"] = np.std(vals)

        logger.info(f"\n{'='*60}")
        logger.info(f"Level {level} Summary ({len(all_fold_metrics)} folds):")
        logger.info(f"  Avg F1:  {avg_metrics.get('avg_f1', 0):.4f} ± {avg_metrics.get('std_f1', 0):.4f}")
        logger.info(f"  Avg AUC: {avg_metrics.get('avg_auc', 0):.4f} ± {avg_metrics.get('std_auc', 0):.4f}")
        logger.info(f"  Avg P:   {avg_metrics.get('avg_precision', 0):.4f}")
        logger.info(f"  Avg R:   {avg_metrics.get('avg_recall', 0):.4f}")
        logger.info(f"{'='*60}")

        summary_path = Path(cfg["output"]["dir"]) / level / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "level": level,
                "description": level_cfg["description"],
                "n_folds": len(all_fold_metrics),
                "avg_metrics": avg_metrics,
                "fold_metrics": all_fold_metrics,
            }, f, indent=2, default=str)

        return avg_metrics
    return {}


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/finetune_config.yaml")
    parser.add_argument("--level", type=str, default="all", choices=["L0", "L1", "L2", "L3", "all"])
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    cfg = load_config(args.config)

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    cases = load_cases(cfg["data"]["cases_dir"])
    if not cases:
        logger.error("No valid cases found!")
        return

    levels = ["L0", "L1", "L2", "L3"] if args.level == "all" else [args.level]

    needs_cot = any(
        cfg["levels"][lv]["loss_weights"].get("cot_distill", 0) > 0
        for lv in levels
    )
    cot_embeddings = {}
    if needs_cot:
        cot_path = cfg["data"].get("cot_results")
        if cot_path and Path(cot_path).exists():
            cot_model = cfg.get("cot_distill", {}).get(
                "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"
            )
            cot_embeddings = load_cot_embeddings(cot_path, cot_model)
    results = {}

    for level in levels:
        metrics = run_level(level, cfg, cases, cot_embeddings)
        results[level] = metrics

        if level != levels[-1] and len(levels) > 1:
            prev_f1 = results.get(levels[levels.index(level) - 1], {}).get("avg_f1", 0) if levels.index(level) > 0 else 0
            curr_f1 = metrics.get("avg_f1", 0)
            if levels.index(level) > 0 and curr_f1 - prev_f1 < 0.02:
                logger.info(
                    f"F1 improvement {curr_f1 - prev_f1:.4f} < 2% threshold. "
                    f"Stopping at {level}."
                )
                break

    logger.info(f"\n{'='*60}")
    logger.info("Final Comparison:")
    for level, metrics in results.items():
        f1 = metrics.get("avg_f1", 0)
        auc = metrics.get("avg_auc", 0)
        logger.info(f"  {level}: F1={f1:.4f}, AUC={auc:.4f}")
    logger.info(f"{'='*60}")

    final_path = Path(cfg["output"]["dir"]) / "comparison.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {final_path}")


if __name__ == "__main__":
    main()
