"""
Pre-training entry point.

Usage:
    # Round 1: small-scale validation
    python -m scripts.run_pretrain --config configs/pretrain_config.yaml --round 1

    # Round 2: medium-scale hyperparameter tuning
    python -m scripts.run_pretrain --config configs/pretrain_config.yaml --round 2

    # Round 3: full-scale training
    python -m scripts.run_pretrain --config configs/pretrain_config.yaml --round 3

    # All 3 rounds sequentially
    python -m scripts.run_pretrain --config configs/pretrain_config.yaml --round all
"""
import argparse
import sys
import json
from pathlib import Path

import yaml
import torch
from loguru import logger

# project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crypto_dataset import build_dataloaders, CryptoPretrainDataset
from models.ts_moe_encoder import PreTrainingModel
from training.pretrain_trainer import PreTrainTrainer


def setup_logging(output_dir: str):
    log_file = Path(output_dir) / "pretrain.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_file), level="DEBUG", rotation="50 MB")


def detect_n_features(cfg: dict) -> int:
    """Detect number of features by loading a small sample."""
    logger.info("Detecting feature count from data...")
    sample_ds = CryptoPretrainDataset(
        data_dir=cfg["data"]["raw_dir"],
        symbols=cfg["data"]["symbols"][:1],
        seq_len=cfg["data"]["seq_len"],
        stride=cfg["data"]["seq_len"],  # no overlap for detection
        anomaly_cfg={
            "price_surge_threshold": cfg["data"]["anomaly"]["price_surge_threshold"],
            "volume_multiplier": cfg["data"]["anomaly"]["volume_multiplier"],
            "surge_window": cfg["data"]["anomaly"]["surge_window"],
            "min_gap": cfg["data"]["anomaly"]["min_gap_between"],
        },
        data_fraction=0.001,
        is_validation=False,
        val_ratio=cfg["data"].get("val_ratio", 0.1),
    )
    if len(sample_ds) == 0:
        raise ValueError("No data found. Check data_dir and symbols in config.")
    n_features = sample_ds.n_features
    logger.info(f"Detected {n_features} features, {len(sample_ds)} sample windows")
    return n_features


def main():
    parser = argparse.ArgumentParser(description="Pre-training entry point")
    parser.add_argument(
        "--config", type=str, default="configs/pretrain_config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--round", type=str, default="1",
        help="Training round: 1, 2, 3, or 'all'"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output"]["dir"]
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Phase 2: Self-supervised Pre-training")
    logger.info("=" * 60)

    # Set seed
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Detect features
    n_features = detect_n_features(cfg)

    # Build model
    model = PreTrainingModel.from_config(cfg["model"], n_features)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params:,} params ({n_trainable:,} trainable)")

    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    # Determine rounds to run
    rounds = cfg["training"]["rounds"]
    if args.round == "all":
        round_indices = list(range(len(rounds)))
    else:
        round_indices = [int(args.round) - 1]

    trainer = PreTrainTrainer(model, cfg, output_dir)

    all_history = {}
    for ridx in round_indices:
        round_cfg = rounds[ridx]
        logger.info(f"\nBuilding dataloaders for {round_cfg['name']}...")

        train_loader, val_loader = build_dataloaders(cfg, round_cfg)
        logger.info(
            f"Train: {len(train_loader.dataset)} windows, "
            f"Val: {len(val_loader.dataset)} windows, "
            f"Anomaly ratio: {train_loader.dataset.get_anomaly_ratio():.3f}"
        )

        history = trainer.train_round(ridx, round_cfg, train_loader, val_loader)
        all_history[round_cfg["name"]] = history

        # For rounds 2, 3: load best checkpoint from previous round
        if ridx < len(rounds) - 1:
            best_ckpt = Path(output_dir) / round_cfg["name"] / "best.pt"
            if best_ckpt.exists():
                logger.info(f"Loading best checkpoint from {round_cfg['name']}")
                ckpt = torch.load(best_ckpt, map_location="cpu")
                model.load_state_dict(ckpt["model_state_dict"])

    # Save final summary
    summary_path = Path(output_dir) / "pretrain_summary.json"
    summary = {
        "n_features": n_features,
        "n_params": n_params,
        "model_config": cfg["model"],
        "rounds_completed": [rounds[i]["name"] for i in round_indices],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nPre-training complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
