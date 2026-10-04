"""
Baseline experiments entry point.

Usage:
    python -m scripts.run_baselines --config configs/pretrain_config.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.traditional_ml import run_baselines


def main():
    parser = argparse.ArgumentParser(description="Baseline ML experiments")
    parser.add_argument(
        "--config", type=str, default="configs/pretrain_config.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/root/project/outputs/baselines",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Traditional ML Baseline Experiments")
    logger.info("=" * 60)

    results = run_baselines(cfg, args.output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Data: {results['data_stats']['n_train']} train, "
                f"{results['data_stats']['n_val']} val, "
                f"{results['data_stats']['n_features']} features")
    logger.info(f"Positive ratio: {results['data_stats']['pos_ratio_train']:.4f}")
    for name in ["rf", "xgb"]:
        m = results[name]
        logger.info(f"\n{m['model']}:")
        logger.info(f"  Accuracy:  {m['accuracy']:.4f}")
        logger.info(f"  Precision: {m['precision']:.4f}")
        logger.info(f"  Recall:    {m['recall']:.4f}")
        logger.info(f"  F1:        {m['f1']:.4f}")
        logger.info(f"  AUC:       {m['auc']:.4f}")
        logger.info(f"  Time:      {m['train_time_s']:.1f}s")


if __name__ == "__main__":
    main()
