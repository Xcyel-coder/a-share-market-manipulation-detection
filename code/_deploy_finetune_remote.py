"""Deploy fine-tuning code + A-share data + CoT embeddings to AutoDL, then run."""
import os
import sys
import tarfile
import time
from pathlib import Path

import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]
REMOTE_BASE = "/root/project"

LOCAL_BASE = Path(__file__).parent

FILES_TO_UPLOAD = {
    "model/models/moe.py": "model/models/moe.py",
    "model/models/ts_moe_encoder.py": "model/models/ts_moe_encoder.py",
    "model/models/finetune_model.py": "model/models/finetune_model.py",
    "model/data/astock_dataset.py": "model/data/astock_dataset.py",
    "model/data/feature_engineering.py": "model/data/feature_engineering.py",
    "model/training/finetune_trainer.py": "model/training/finetune_trainer.py",
    "model/analysis/expert_analysis.py": "model/analysis/expert_analysis.py",
    "model/evaluation/__init__.py": "model/evaluation/__init__.py",
    "model/evaluation/eval_framework.py": "model/evaluation/eval_framework.py",
    "model/evaluation/baselines.py": "model/evaluation/baselines.py",
    "model/evaluation/robustness.py": "model/evaluation/robustness.py",
    "model/evaluation/backtest.py": "model/evaluation/backtest.py",
    "model/outputs/cot/cot_embeddings.npz": "model/outputs/cot/cot_embeddings.npz",
    "model/outputs/cot/cot_results.json": "model/outputs/cot/cot_results.json",
}


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    return ssh


def upload_files(ssh):
    sftp = ssh.open_sftp()

    for local_rel, remote_rel in FILES_TO_UPLOAD.items():
        local_path = LOCAL_BASE / local_rel
        remote_path = f"{REMOTE_BASE}/{remote_rel}"

        if not local_path.exists():
            print(f"  SKIP (not found): {local_rel}")
            continue

        remote_dir = "/".join(remote_path.split("/")[:-1])
        try:
            ssh.exec_command(f"mkdir -p {remote_dir}")
            time.sleep(0.2)
        except Exception:
            pass

        size = local_path.stat().st_size / 1024
        sftp.put(str(local_path), remote_path)
        print(f"  OK: {local_rel} ({size:.0f} KB)")

    sftp.close()


def upload_astock_data(ssh):
    """Tar and upload A-share data."""
    astock_dir = LOCAL_BASE / "data_pipeline" / "data" / "raw" / "astock"
    cases_dir = LOCAL_BASE / "data_pipeline" / "data" / "processed" / "extracted_cases"

    tar_path = os.path.join(os.environ.get("TEMP", "/tmp"), "astock_data.tar.gz")

    print("  Creating tar.gz of A-share data...")
    with tarfile.open(tar_path, "w:gz") as tar:
        for parquet in sorted(astock_dir.rglob("*.parquet")):
            arcname = f"astock/{parquet.parent.name}/{parquet.name}"
            tar.add(str(parquet), arcname=arcname)

        for jf in sorted(cases_dir.glob("*.json")):
            arcname = f"extracted_cases/{jf.name}"
            tar.add(str(jf), arcname=arcname)

    size_mb = os.path.getsize(tar_path) / 1024 / 1024
    print(f"  Tar size: {size_mb:.1f} MB")

    sftp = ssh.open_sftp()
    sftp.put(tar_path, f"{REMOTE_BASE}/astock_data.tar.gz")
    sftp.close()
    print("  Upload complete")

    print("  Extracting on remote...")
    cmd = (
        f"cd {REMOTE_BASE} && "
        f"mkdir -p data/raw data/processed && "
        f"cd data/raw && tar xzf {REMOTE_BASE}/astock_data.tar.gz && "
        f"mv extracted_cases ../processed/ 2>/dev/null; "
        f"rm {REMOTE_BASE}/astock_data.tar.gz && "
        f"echo 'Astock files:' && find astock -name '*.parquet' | wc -l && "
        f"echo 'Case files:' && find ../processed/extracted_cases -name '*.json' 2>/dev/null | wc -l"
    )
    _, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    print(stdout.read().decode())
    os.remove(tar_path)


def create_remote_runner(ssh):
    """Create the fine-tuning runner script on remote."""
    script = '''#!/usr/bin/env python3
"""Remote fine-tuning with real pre-trained weights on AutoDL RTX 5090."""
import sys, json, time, numpy as np, torch
from pathlib import Path

sys.path.insert(0, "/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/root/project/outputs/finetune/finetune_remote.log", level="DEBUG")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from training.finetune_trainer import FineTuneTrainer
from torch.utils.data import DataLoader

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
OUTPUT_DIR = "/root/project/outputs/finetune"
PRETRAINED = "/root/project/outputs/pretrain/round3_full/best.pt"
COT_CACHE = "/root/project/model/outputs/cot/cot_embeddings.npz"

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

def run_level(level, cases, cot_embeddings):
    level_cfg = LEVELS[level]
    logger.info(f"\\n{'='*60}")
    logger.info(f"Level {level}: {level_cfg['description']}")
    logger.info(f"Using pretrained: {PRETRAINED}")
    logger.info(f"{'='*60}")

    splits = build_cv_splits(cases, n_folds=5, seed=42)
    all_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        logger.info(f"\\n--- Fold {fold_idx+1}/5 (train={len(train_cases)}, val={len(val_cases)}) ---")

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
            PRETRAINED, MODEL_CFG, n_features=n_features,
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
        logger.info(f"\\n{level} Summary: F1={avg.get('f1',0):.4f} AUC={avg.get('auc',0):.4f} "
                     f"P={avg.get('precision',0):.4f} R={avg.get('recall',0):.4f}")
        return avg
    return {}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Pretrained weights: {PRETRAINED}")

    cases = load_cases()
    logger.info(f"Loaded {len(cases)} valid cases")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load CoT embeddings
    cot_embeddings = {}
    if Path(COT_CACHE).exists():
        data = np.load(COT_CACHE)
        cot_embeddings = {k: data[k] for k in data.files}
        logger.info(f"Loaded CoT embeddings for {len(cot_embeddings)} cases")

    results = {}
    for level in ["L0", "L1", "L2", "L3"]:
        t0 = time.time()
        metrics = run_level(level, cases, cot_embeddings)
        elapsed = time.time() - t0
        results[level] = metrics
        logger.info(f"{level} completed in {elapsed:.1f}s")

    logger.info(f"\\n{'='*60}")
    logger.info("Final Comparison (with pretrained weights):")
    for lv, m in results.items():
        logger.info(f"  {lv}: F1={m.get('f1',0):.4f} AUC={m.get('auc',0):.4f}")
    logger.info(f"{'='*60}")

    with open(Path(OUTPUT_DIR) / "comparison_pretrained.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("All done! Results saved to comparison_pretrained.json")
'''

    _, stdout, _ = ssh.exec_command(f"cat > {REMOTE_BASE}/run_finetune_remote.py << 'PYEOF'\n{script}\nPYEOF")
    stdout.read()
    print("  Remote runner script created")


def install_remote_deps(ssh):
    """Install dependencies on remote."""
    cmd = (
        "source /etc/profile && "
        "pip install scikit-learn loguru pyarrow -q 2>&1 | tail -3"
    )
    _, stdout, stderr = ssh.exec_command(cmd, timeout=120)
    print(stdout.read().decode())


def main():
    print("=" * 60)
    print("Deploying fine-tuning to AutoDL RTX 5090")
    print("=" * 60)

    print("\n[1/5] Connecting to AutoDL...")
    ssh = connect()
    print("  Connected!")

    print("\n[2/5] Uploading model code...")
    upload_files(ssh)

    print("\n[3/5] Uploading A-share data...")
    upload_astock_data(ssh)

    print("\n[4/5] Creating remote runner + installing deps...")
    create_remote_runner(ssh)
    install_remote_deps(ssh)

    print("\n[5/5] Launching fine-tuning (nohup)...")
    cmd = (
        f"source /etc/profile && "
        f"cd {REMOTE_BASE} && "
        f"nohup python run_finetune_remote.py > finetune_stdout.log 2>&1 &"
        f" echo $!"
    )
    _, stdout, _ = ssh.exec_command(cmd)
    pid = stdout.read().decode().strip()
    print(f"  Started! PID={pid}")
    print(f"  Monitor: ssh root@{HOST} -p {PORT} 'tail -f {REMOTE_BASE}/finetune_stdout.log'")

    ssh.close()
    print("\nDeployment complete! Fine-tuning running on AutoDL.")
    print("Use _monitor_remote.py to check progress.")


if __name__ == "__main__":
    main()
