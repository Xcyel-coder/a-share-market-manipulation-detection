#!/usr/bin/env python3
"""
Expanded fine-tuning: Uses scale-aligned pretrain checkpoint + expanded dataset (126 cases).
L0-L3 progressive fine-tuning with 5-fold CV.
"""
import sys, json, os, time, torch, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")
os.chdir("/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/root/project/outputs/expanded_finetune/rerun.log", rotation="50MB", level="DEBUG")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from torch.utils.data import DataLoader

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
COT_PATH = "/root/project/model/outputs/cot/cot_results.json"
PRETRAIN_CKPT = "/root/project/outputs/pretrain_daily/align_daily/best.pt"
OUTPUT_DIR = Path("/root/project/outputs/expanded_finetune")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result


# ============ Load ALL cases (original 23 + new 103) ============
cases = []
loaded_ids = set()

for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
        doc_id = c.get("doc_id", "")
        code = c.get("stock_code", "")
        start = c.get("manipulation_start_date")
        end = c.get("manipulation_end_date")

        if not code or not start or not end:
            logger.debug(f"  Skip (no code/dates): {doc_id}")
            continue

        astock_path = Path(ASTOCK_DIR) / doc_id / "daily.parquet"
        if not astock_path.exists():
            logger.debug(f"  Skip (no data): {doc_id}")
            continue

        if doc_id in loaded_ids:
            logger.debug(f"  Skip (duplicate): {doc_id}")
            continue

        cases.append(c)
        loaded_ids.add(doc_id)
        logger.debug(f"  Loaded: {doc_id} ({code})")

logger.info(f"Loaded {len(cases)} valid cases ({len([c for c in cases if c['doc_id'].startswith('CSRC')])} original + {len([c for c in cases if c['doc_id'].startswith('CASE')])} new)")

# Load CoT embeddings (only available for original cases)
cot_embeddings = {}
COT_NPZ = "/root/project/model/outputs/cot/cot_embeddings.npz"
if Path(COT_NPZ).exists():
    npz_data = np.load(COT_NPZ)
    cot_embeddings = {k: npz_data[k] for k in npz_data.files}
    logger.info(f"CoT embeddings loaded: {len(cot_embeddings)}")
else:
    logger.warning("No CoT embeddings found")

splits = build_cv_splits(cases, n_folds=5, seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

from training.finetune_trainer import FocalLoss

LEVELS = {
    "L0": {"custom_router": False, "expert_freeze": False},
    "L2": {"custom_router": True, "metadata_dim": 8, "expert_freeze": False},
}
LOSS_WEIGHTS = {
    "L0": {"classification": 1.0, "cot_distill": 0.0},
    "L2": {"classification": 0.7, "cot_distill": 0.3},
}

EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 32
PATIENCE = 10

all_results = {}

for level_name, level_cfg in LEVELS.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"Level: {level_name}")
    logger.info(f"{'='*60}")

    level_dir = OUTPUT_DIR / level_name
    level_dir.mkdir(exist_ok=True)

    fold_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        fold_dir = level_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        logger.info(f"\n--- Fold {fold_idx} ({len(train_cases)} train, {len(val_cases)} val) ---")

        train_ds = AStockFinetuneDataset(
            cases=train_cases,
            astock_dir=ASTOCK_DIR,
            feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30,
            overlap_ratio=0.5,
            oversample_positive=1,
            cot_embeddings=cot_embeddings,
        )
        val_ds = AStockFinetuneDataset(
            cases=val_cases,
            astock_dir=ASTOCK_DIR,
            feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30,
            overlap_ratio=0.5,
            oversample_positive=1,
            cot_embeddings=cot_embeddings,
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning(f"  Fold {fold_idx}: empty dataset, skip")
            continue

        n_features = train_ds.get_n_features()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=custom_collate, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=custom_collate, num_workers=2, pin_memory=True)

        model = FineTuneModel.from_pretrained(
            checkpoint_path=PRETRAIN_CKPT,
            model_cfg=MODEL_CFG,
            n_features=n_features,
            level=level_name,
            level_cfg=level_cfg,
            cot_cfg={"embedding_dim": 384, "projection_dim": 128},
        )
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = EPOCHS * len(train_loader)
        warmup = int(total_steps * 0.1)
        sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [sched_warmup, sched_cosine], milestones=[warmup])

        loss_w = LOSS_WEIGHTS[level_name]
        n_pos = sum(1 for w in train_ds.windows if w["label"] == 1)
        n_neg = len(train_ds) - n_pos
        alpha = n_pos / max(len(train_ds), 1)
        focal = FocalLoss(gamma=2.0, alpha=alpha).to(device)

        best_val_mcc = -1.0
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                labels = batch_gpu["label"]

                out = model(batch_gpu)
                loss = loss_w["classification"] * focal(out["logits"], labels)
                if loss_w["cot_distill"] > 0 and "cls_proj" in out and "cot_target_proj" in out:
                    has_cot = batch_gpu.get("has_cot")
                    cot_loss = torch.nn.functional.mse_loss(
                        out["cls_proj"], out["cot_target_proj"], reduction="none")
                    if has_cot is not None:
                        cot_loss = (cot_loss * has_cot.unsqueeze(-1)).mean()
                    else:
                        cot_loss = cot_loss.mean()
                    loss = loss + loss_w["cot_distill"] * cot_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            # Validate
            model.eval()
            all_preds, all_labels = [], []
            val_loss_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                    labels = batch_gpu["label"]
                    out = model(batch_gpu)
                    vloss = focal(out["logits"], labels)
                    val_loss_total += vloss.item()
                    preds = out["logits"].argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            val_loss = val_loss_total / max(len(val_loader), 1)
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            mcc = float(tp*tn - fp*fn) / denom if denom > 0 else 0.0
            acc = (y_pred == y_true).mean()

            if (epoch + 1) % 5 == 0 or mcc > best_val_mcc:
                logger.info(f"  [{level_name}] Fold{fold_idx} Epoch {epoch+1}/{EPOCHS} | "
                           f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                           f"MCC: {mcc:.4f} | Acc: {acc:.4f} | "
                           f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}")

            if mcc > best_val_mcc:
                best_val_mcc = mcc
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "mcc": mcc,
                    "metrics": {"mcc": mcc, "acc": acc, "tp": int(tp), "tn": int(tn),
                                "fp": int(fp), "fn": int(fn)},
                }, fold_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        fold_metrics.append({"fold": fold_idx, "best_mcc": best_val_mcc})
        logger.info(f"  Fold {fold_idx} best MCC: {best_val_mcc:.4f}")

    avg_mcc = np.mean([m["best_mcc"] for m in fold_metrics]) if fold_metrics else 0
    std_mcc = np.std([m["best_mcc"] for m in fold_metrics]) if fold_metrics else 0
    all_results[level_name] = {
        "fold_metrics": fold_metrics,
        "avg_mcc": float(avg_mcc),
        "std_mcc": float(std_mcc),
    }
    logger.info(f"\n{level_name} 5-fold CV: MCC = {avg_mcc:.4f} ± {std_mcc:.4f}")

# Save results
with open(OUTPUT_DIR / "all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

logger.info(f"\n{'='*60}")
logger.info("FINAL RESULTS (Scale-Aligned Pretrain + Expanded Dataset)")
logger.info(f"{'='*60}")
for level, res in all_results.items():
    logger.info(f"  {level}: MCC = {res['avg_mcc']:.4f} ± {res['std_mcc']:.4f}")
logger.info(f"\nResults saved to {OUTPUT_DIR / 'all_results.json'}")
