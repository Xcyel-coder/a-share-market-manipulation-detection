"""
Scale-aligned pre-training entry point.

Loads the existing minute-level pretrain checkpoint and runs additional
alignment rounds on daily/4h aggregated crypto data, using the same
25-feature pipeline as A-stock fine-tuning.

Usage:
    python -m scripts.run_pretrain_daily --config configs/pretrain_daily_config.yaml
"""
import argparse
import sys
import json
from pathlib import Path

import yaml
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crypto_daily_dataset import CryptoDailyPretrainDataset, build_daily_dataloaders
from models.ts_moe_encoder import PreTrainingModel
from training.pretrain_trainer import PreTrainTrainer


def setup_logging(output_dir: str):
    log_file = Path(output_dir) / "pretrain_daily.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_file), level="DEBUG", rotation="50 MB")


def detect_n_features(cfg: dict) -> int:
    """Detect number of features from daily aggregated data."""
    data_cfg = cfg["data"]
    logger.info("Detecting feature count from daily data...")
    sample_ds = CryptoDailyPretrainDataset(
        data_dir=data_cfg["aggregated_dir"],
        symbols=data_cfg["symbols"][:1],
        timeframe="daily",
        seq_len=data_cfg["seq_len"],
        stride=data_cfg["seq_len"],
        anomaly_cfg={
            "price_surge_threshold": 0.05,
            "volume_multiplier": 2.0,
            "surge_window": 3,
            "min_gap": 5,
        },
        feature_cfg=data_cfg.get("feature_params", {}),
        data_fraction=1.0,
        is_validation=False,
        val_ratio=data_cfg.get("val_ratio", 0.1),
    )
    if len(sample_ds) == 0:
        raise ValueError("No daily data found. Check aggregated_dir in config.")
    n_features = sample_ds.n_features
    logger.info(f"Detected {n_features} features (daily), "
                f"{len(sample_ds)} sample windows, "
                f"feature_names={sample_ds.feature_names[:5]}...")
    return n_features


def main():
    parser = argparse.ArgumentParser(description="Scale-aligned pre-training")
    parser.add_argument(
        "--config", type=str, default="configs/pretrain_daily_config.yaml",
    )
    parser.add_argument("--round", type=str, default="all")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output"]["dir"]
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Scale-Aligned Pre-training (Daily + 4H)")
    logger.info("=" * 60)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    n_features = detect_n_features(cfg)

    model = PreTrainingModel.from_config(cfg["model"], n_features)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params")

    # Load existing minute-level pretrain checkpoint
    resume_path = cfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        logger.info(f"Loading minute-level pretrain checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

        # The checkpoint has model_state_dict from PreTrainingModel trained
        # with minute-level features (different n_features). We need to load
        # only the encoder weights that are dimension-compatible.
        state = ckpt.get("model_state_dict", ckpt)
        model_state = model.state_dict()

        loaded, skipped = 0, 0
        compatible_state = {}
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                compatible_state[k] = v
                loaded += 1
            else:
                skipped += 1

        model.load_state_dict(compatible_state, strict=False)
        logger.info(f"Loaded {loaded} params, skipped {skipped} (shape mismatch)")
        logger.info("Input projection and task heads will be re-initialized for 25-dim features")
    else:
        logger.info("No pretrain checkpoint found, training from scratch")

    rounds = cfg["training"]["rounds"]
    if args.round == "all":
        round_indices = list(range(len(rounds)))
    else:
        round_indices = [int(args.round) - 1]

    trainer = PreTrainTrainer(model, cfg, output_dir)

    for ridx in round_indices:
        round_cfg = rounds[ridx]
        timeframe = round_cfg.get("timeframe", "daily")
        logger.info(f"\n{'='*60}")
        logger.info(f"Alignment Round: {round_cfg['name']} (timeframe={timeframe})")
        logger.info(f"{'='*60}")

        train_loader, val_loader = build_daily_dataloaders(cfg, round_cfg)
        logger.info(
            f"Train: {len(train_loader.dataset)} windows, "
            f"Val: {len(val_loader.dataset)} windows, "
            f"Anomaly ratio: {train_loader.dataset.get_anomaly_ratio():.3f}"
        )

        if len(train_loader.dataset) == 0:
            logger.warning(f"No data for {timeframe}, skipping")
            continue

        history = trainer.train_round(ridx, round_cfg, train_loader, val_loader)

        best_ckpt = Path(output_dir) / round_cfg["name"] / "best.pt"
        if best_ckpt.exists() and ridx < len(rounds) - 1:
            logger.info(f"Loading best from {round_cfg['name']} for next round")
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

    summary_path = Path(output_dir) / "pretrain_daily_summary.json"
    summary = {
        "n_features": n_features,
        "n_params": n_params,
        "model_config": cfg["model"],
        "rounds_completed": [rounds[i]["name"] for i in round_indices],
        "resume_from": str(resume_path),
        "strategy": "scale_aligned_daily_4h",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nScale-aligned pre-training complete! Summary: {summary_path}")


if __name__ == "__main__":
    main()
