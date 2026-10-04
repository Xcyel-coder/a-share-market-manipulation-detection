#!/usr/bin/env python3
"""
Ablation study to isolate the effects of:
  A) Scale-aligned pretrain (23 original cases + daily-aligned pretrain)
  B) Dataset expansion (126 cases + original minute pretrain)
Compares against:
  - Original: 23 cases + minute pretrain (already done, MCC ~0.88)
  - Full new: 126 cases + daily-aligned pretrain (done, MCC ~0.37)
"""
import sys, json, os, time, torch, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")
os.chdir("/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/root/project/outputs/ablation_study/ablation.log", rotation="50MB", level="DEBUG")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from torch.utils.data import DataLoader
from training.finetune_trainer import FocalLoss

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"

DAILY_ALIGNED_CKPT = "/root/project/outputs/pretrain_daily/align_daily/best.pt"
ORIGINAL_MINUTE_CKPT = "/root/project/outputs/pretrain/round3_full/best.pt"

OUTPUT_DIR = Path("/root/project/outputs/ablation_study")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_ORIGINAL_IDS = [
    'CSRC-2017-064', 'CSRC-2017-077', 'CSRC-2017-098', 'CSRC-2018-002',
    'CSRC-2018-012', 'CSRC-2018-014', 'CSRC-2018-022', 'CSRC-2021-124',
    'CSRC-2022-064', 'CSRC-2023-005', 'CSRC-2023-060', 'CSRC-2023-077',
    'CSRC-2023-083', 'CSRC-2024-008', 'CSRC-2024-023', 'CSRC-2024-058',
    'CSRC-2024-059', 'CSRC-2024-082', 'CSRC-2025-003', 'CSRC-URL-7532746',
    'CSRC-URL-7532795', 'CSRC-URL-7564434', 'CSRC-URL-7610753',
]

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


def load_cases(filter_ids=None):
    """Load cases, optionally filtering to specific IDs."""
    cases = []
    loaded = set()
    for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
        with open(f, encoding="utf-8") as fh:
            c = json.load(fh)
        doc_id = c.get("doc_id", "")
        code = c.get("stock_code", "")
        start = c.get("manipulation_start_date")
        end = c.get("manipulation_end_date")
        if not code or not start or not end:
            continue
        if filter_ids is not None and doc_id not in filter_ids:
            continue
        astock_path = Path(ASTOCK_DIR) / doc_id / "daily.parquet"
        if not astock_path.exists():
            continue
        if doc_id in loaded:
            continue
        cases.append(c)
        loaded.add(doc_id)
    return cases


def run_experiment(exp_name, cases, pretrain_ckpt, n_folds=5):
    """Run L0 fine-tuning with given cases and pretrain checkpoint."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Cases: {len(cases)}, Pretrain: {Path(pretrain_ckpt).name}")
    logger.info(f"{'='*60}")

    exp_dir = OUTPUT_DIR / exp_name
    exp_dir.mkdir(exist_ok=True)

    cot_embeddings = {}
    COT_NPZ = "/root/project/model/outputs/cot/cot_embeddings.npz"
    if Path(COT_NPZ).exists():
        npz_data = np.load(COT_NPZ)
        cot_embeddings = {k: npz_data[k] for k in npz_data.files}

    splits = build_cv_splits(cases, n_folds=n_folds, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        fold_dir = exp_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        logger.info(f"\n--- Fold {fold_idx} ({len(train_cases)} train, {len(val_cases)} val) ---")

        train_ds = AStockFinetuneDataset(
            cases=train_cases, astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings=cot_embeddings)
        val_ds = AStockFinetuneDataset(
            cases=val_cases, astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings=cot_embeddings)

        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        n_features = train_ds.get_n_features()
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  collate_fn=custom_collate, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                                collate_fn=custom_collate, num_workers=2, pin_memory=True)

        model = FineTuneModel.from_pretrained(
            checkpoint_path=pretrain_ckpt,
            model_cfg=MODEL_CFG, n_features=n_features,
            level="L0", level_cfg={"custom_router": False, "expert_freeze": False},
            cot_cfg={"embedding_dim": 384, "projection_dim": 128})
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        total_steps = 50 * len(train_loader)
        warmup = int(total_steps * 0.1)
        sched_w = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_w, sched_c], milestones=[warmup])

        n_pos = sum(1 for w in train_ds.windows if w["label"] == 1)
        alpha = n_pos / max(len(train_ds), 1)
        focal = FocalLoss(gamma=2.0, alpha=alpha).to(device)

        best_mcc, patience = -1.0, 0
        for epoch in range(50):
            model.train()
            t_loss = 0
            for batch in train_loader:
                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch_gpu)
                loss = focal(out["logits"], batch_gpu["label"])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                t_loss += loss.item()
            t_loss /= max(len(train_loader), 1)

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    out = model(batch_gpu)
                    preds_all.extend(out["logits"].argmax(-1).cpu().numpy())
                    labels_all.extend(batch_gpu["label"].cpu().numpy())

            y_t, y_p = np.array(labels_all), np.array(preds_all)
            tp = ((y_p==1)&(y_t==1)).sum(); tn = ((y_p==0)&(y_t==0)).sum()
            fp = ((y_p==1)&(y_t==0)).sum(); fn = ((y_p==0)&(y_t==1)).sum()
            d = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            mcc = float(tp*tn-fp*fn)/d if d>0 else 0.0

            if mcc > best_mcc:
                best_mcc = mcc; patience = 0
                torch.save({"model_state_dict": model.state_dict(), "mcc": mcc,
                             "epoch": epoch}, fold_dir / "best.pt")
                logger.info(f"  [{exp_name}] Fold{fold_idx} Ep{epoch+1} MCC={mcc:.4f} (new best)")
            else:
                patience += 1
                if patience >= 10:
                    logger.info(f"  Early stop ep {epoch+1}")
                    break

            if (epoch+1) % 10 == 0:
                logger.info(f"  [{exp_name}] Fold{fold_idx} Ep{epoch+1} MCC={mcc:.4f} loss={t_loss:.4f}")

        fold_metrics.append({"fold": fold_idx, "best_mcc": best_mcc})
        logger.info(f"  Fold {fold_idx} best MCC: {best_mcc:.4f}")

    avg = np.mean([m["best_mcc"] for m in fold_metrics])
    std = np.std([m["best_mcc"] for m in fold_metrics])
    result = {"fold_metrics": fold_metrics, "avg_mcc": float(avg), "std_mcc": float(std)}
    logger.info(f"\n{exp_name}: MCC = {avg:.4f} ± {std:.4f}")
    return result


# ============ Run ablation experiments ============
all_results = {}

# Ablation A: 23 original cases + daily-aligned pretrain
cases_23 = load_cases(filter_ids=set(VALID_ORIGINAL_IDS))
logger.info(f"Loaded {len(cases_23)} original cases")
all_results["A_23cases_daily_pretrain"] = run_experiment(
    "A_23cases_daily_pretrain", cases_23, DAILY_ALIGNED_CKPT)

# Ablation B: 126 expanded cases + original minute pretrain
cases_all = load_cases(filter_ids=None)
logger.info(f"Loaded {len(cases_all)} expanded cases")
all_results["B_126cases_minute_pretrain"] = run_experiment(
    "B_126cases_minute_pretrain", cases_all, ORIGINAL_MINUTE_CKPT)

# Save summary
with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

logger.info(f"\n{'='*60}")
logger.info("ABLATION STUDY COMPLETE")
logger.info(f"{'='*60}")
logger.info(f"A (23 cases + aligned pretrain): MCC = {all_results['A_23cases_daily_pretrain']['avg_mcc']:.4f}")
logger.info(f"B (126 cases + minute pretrain): MCC = {all_results['B_126cases_minute_pretrain']['avg_mcc']:.4f}")
logger.info(f"Reference - Original (23 cases + minute pretrain): MCC ≈ 0.8829")
logger.info(f"Reference - Full new (126 cases + aligned pretrain): MCC ≈ 0.3736")
