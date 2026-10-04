"""Deploy and restart L1-L3 fine-tuning on AutoDL (L0 already complete)."""
import os
import paramiko
import time

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)


def run(cmd, timeout=30):
    print(f">>> {cmd}")
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out.strip():
        print(out.strip())
    if err.strip():
        print(f"[stderr] {err.strip()}")
    return out.strip()


# 1. Kill any lingering python processes
print("=" * 60)
print("Step 1: Clean up old processes")
run("pkill -f run_finetune_remote || true")
time.sleep(2)

# 2. Clean up incomplete L1 results (keep L0 intact)
print("\n" + "=" * 60)
print("Step 2: Clean incomplete L1 (keep fold_0, remove incomplete fold_1)")
run("rm -rf /root/project/outputs/finetune/L1/fold_1")
run("rm -rf /root/project/outputs/finetune/L2")
run("rm -rf /root/project/outputs/finetune/L3")

# 3. Verify L0 results are intact
print("\n" + "=" * 60)
print("Step 3: Verify L0 intact")
run("ls -la /root/project/outputs/finetune/L0/*/best.pt 2>/dev/null | wc -l")

# 4. Verify pretrained weights exist
print("\n" + "=" * 60)
print("Step 4: Verify pretrained weights")
run("ls -la /root/project/outputs/pretrain/round3_full/best.pt")

# 5. Verify CoT embeddings
run("ls -la /root/project/model/outputs/cot/cot_embeddings.npz 2>/dev/null || echo 'CoT NOT FOUND'")

# 6. Create new runner script (L1-L3 only)
print("\n" + "=" * 60)
print("Step 5: Create L1-L3 runner script")

script = r'''#!/usr/bin/env python3
"""Fine-tune L1, L2, L3 on AutoDL with real pre-trained weights. L0 already complete."""
import sys, json, time, numpy as np, torch
from pathlib import Path

sys.path.insert(0, "/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/root/project/outputs/finetune/finetune_l1l3.log", level="DEBUG")

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
    logger.info(f"\n{'='*60}")
    logger.info(f"Level {level}: {level_cfg['description']}")
    logger.info(f"Using pretrained: {PRETRAINED}")
    logger.info(f"{'='*60}")

    splits = build_cv_splits(cases, n_folds=5, seed=42)
    all_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx+1}/5 (train={len(train_cases)}, val={len(val_cases)}) ---")

        cot_emb = cot_embeddings if level_cfg["loss_weights"].get("cot_distill", 0) > 0 else {}

        train_ds = AStockFinetuneDataset(
            train_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1,
            cot_embeddings=cot_emb,
        )
        val_ds = AStockFinetuneDataset(
            val_cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
            min_bars=30, overlap_ratio=0.5, oversample_positive=1,
            cot_embeddings=cot_emb,
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
        logger.info(f"\n{level} Summary: F1={avg.get('f1',0):.4f} AUC={avg.get('auc',0):.4f} "
                     f"P={avg.get('precision',0):.4f} R={avg.get('recall',0):.4f}")
        return avg
    return {}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Pretrained weights: {PRETRAINED}")
    logger.info(f"NOTE: L0 already complete, running L1/L2/L3 only")

    cases = load_cases()
    logger.info(f"Loaded {len(cases)} valid cases")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cot_embeddings = {}
    if Path(COT_CACHE).exists():
        data = np.load(COT_CACHE)
        cot_embeddings = {k: data[k] for k in data.files}
        logger.info(f"Loaded CoT embeddings for {len(cot_embeddings)} cases")
    else:
        logger.warning(f"CoT cache not found: {COT_CACHE}")

    # Load existing L0 results
    l0_result_path = Path(OUTPUT_DIR) / "L0_summary.json"
    results = {}
    if l0_result_path.exists():
        results["L0"] = json.load(open(l0_result_path))
        logger.info(f"Loaded L0 results from cache")
    else:
        # Reconstruct L0 from individual fold histories
        l0_metrics = []
        for fold in range(5):
            hp = Path(OUTPUT_DIR) / "L0" / f"fold_{fold}" / "history.json"
            if hp.exists():
                h = json.load(open(hp))
                if h.get("best_metrics"):
                    l0_metrics.append(h["best_metrics"])
        if l0_metrics:
            avg = {k: np.mean([m[k] for m in l0_metrics if isinstance(m.get(k), (int, float))])
                   for k in l0_metrics[0] if isinstance(l0_metrics[0].get(k), (int, float))}
            results["L0"] = avg
            logger.info(f"Reconstructed L0: F1={avg.get('f1',0):.4f} AUC={avg.get('auc',0):.4f}")

    # Run L1, L2, L3
    for level in ["L1", "L2", "L3"]:
        t0 = time.time()
        metrics = run_level(level, cases, cot_embeddings)
        elapsed = time.time() - t0
        results[level] = metrics
        logger.info(f"{level} completed in {elapsed:.1f}s")

    # Save final comparison
    logger.info(f"\n{'='*60}")
    logger.info("Final Comparison (with pretrained weights):")
    for lv in ["L0", "L1", "L2", "L3"]:
        m = results.get(lv, {})
        logger.info(f"  {lv}: F1={m.get('f1',0):.4f} AUC={m.get('auc',0):.4f} "
                     f"P={m.get('precision',0):.4f} R={m.get('recall',0):.4f}")
    logger.info(f"{'='*60}")

    with open(Path(OUTPUT_DIR) / "comparison_pretrained.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("All done! Results saved to comparison_pretrained.json")
'''

sftp = ssh.open_sftp()
with sftp.file("/root/project/run_l1l3.py", "w") as f:
    f.write(script)
sftp.close()
print("Runner script created.")

# 7. Launch via nohup
print("\n" + "=" * 60)
print("Step 6: Launch training")
run("cd /root/project && nohup python -u run_l1l3.py > l1l3_stdout.log 2>&1 &", timeout=5)
time.sleep(3)

# 8. Verify it's running
print("\n" + "=" * 60)
print("Step 7: Verify process started")
run("ps aux | grep run_l1l3 | grep -v grep")
run("cat /root/project/l1l3_stdout.log 2>/dev/null | head -10")

ssh.close()
print("\nDeployment complete. Training L1-L3 in background on AutoDL.")
