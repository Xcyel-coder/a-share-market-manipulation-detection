#!/usr/bin/env python3
"""
Comprehensive experiment on expanded 126-case A-share manipulation dataset.
Runs: MoE L0-L3 (optimized) + all baselines (RF, XGBoost, SVM, LSTM, Transformer)
All share the same 5-fold CV splits for fair comparison.
"""
import sys, json, os, time, torch, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")
os.chdir("/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/root/project/outputs/comprehensive/experiment.log", rotation="50MB", level="DEBUG")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from evaluation.baselines import (
    flatten_windows, LSTMClassifier, TransformerClassifier,
)
from training.finetune_trainer import FocalLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import sklearn.metrics as skm

CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
COT_NPZ = "/root/project/model/outputs/cot/cot_embeddings.npz"
PRETRAIN_DAILY = "/root/project/outputs/pretrain_daily/align_daily/best.pt"
PRETRAIN_MINUTE = "/root/project/outputs/pretrain/round3_full/best.pt"
OUTPUT_DIR = Path("/root/project/outputs/comprehensive")
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

# ==================================================================
# Utilities
# ==================================================================

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


def compute_all_metrics(y_true, y_pred, y_prob=None):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = float(tp*tn - fp*fn) / denom if denom > 0 else 0.0

    precision = float(tp) / max(tp + fp, 1)
    recall = float(tp) / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = float(tp + tn) / max(tp + fp + fn + tn, 1)
    specificity = float(tn) / max(tn + fp, 1)

    metrics = {
        "mcc": mcc, "f1": f1, "accuracy": accuracy,
        "precision": precision, "recall": recall, "specificity": specificity,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(skm.roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.5
        try:
            metrics["pr_auc"] = float(skm.average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = 0.0

    return metrics


def make_balanced_sampler(dataset):
    labels = [w["label"] for w in dataset.windows]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


# ==================================================================
# Load all cases
# ==================================================================
logger.info("Loading cases...")
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
            continue
        astock_path = Path(ASTOCK_DIR) / doc_id / "daily.parquet"
        if not astock_path.exists():
            continue
        if doc_id in loaded_ids:
            continue
        cases.append(c)
        loaded_ids.add(doc_id)

n_orig = len([c for c in cases if c['doc_id'].startswith('CSRC')])
n_new = len([c for c in cases if c['doc_id'].startswith('CASE')])
logger.info(f"Loaded {len(cases)} cases ({n_orig} original + {n_new} new)")

cot_embeddings = {}
if Path(COT_NPZ).exists():
    npz_data = np.load(COT_NPZ)
    cot_embeddings = {k: npz_data[k] for k in npz_data.files}
    logger.info(f"CoT embeddings: {len(cot_embeddings)} cases")

splits = build_cv_splits(cases, n_folds=5, seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}, Folds: {len(splits)}")

# ==================================================================
# Part 1: MoE Model L0-L3 (optimized with balanced sampling)
# ==================================================================

LEVELS = {
    "L0": {"custom_router": False, "expert_freeze": False},
    "L1": {"custom_router": False, "expert_freeze": False},
    "L2": {"custom_router": True, "metadata_dim": 8, "expert_freeze": False},
    "L3": {"custom_router": True, "metadata_dim": 8, "expert_freeze": True, "freeze_top_n_experts": 4},
}
LOSS_WEIGHTS = {
    "L0": {"classification": 1.0, "cot_distill": 0.0},
    "L1": {"classification": 0.7, "cot_distill": 0.3},
    "L2": {"classification": 0.7, "cot_distill": 0.3},
    "L3": {"classification": 0.7, "cot_distill": 0.3},
}

EPOCHS = 60
LR = 5e-5
BATCH_SIZE = 32
PATIENCE = 15

moe_results = {}

for level_name, level_cfg in LEVELS.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"MoE-{level_name} (daily-aligned pretrain)")
    logger.info(f"{'='*60}")

    level_dir = OUTPUT_DIR / f"MoE_{level_name}"
    level_dir.mkdir(exist_ok=True)
    fold_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx} ({len(train_cases)} train, {len(val_cases)} val) ---")
        fold_dir = level_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        train_ds = AStockFinetuneDataset(
            cases=train_cases, astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings=cot_embeddings,
        )
        val_ds = AStockFinetuneDataset(
            cases=val_cases, astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings=cot_embeddings,
        )
        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning(f"  Empty dataset, skip fold {fold_idx}")
            continue

        n_features = train_ds.get_n_features()
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  collate_fn=custom_collate, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=custom_collate, num_workers=2, pin_memory=True)

        model = FineTuneModel.from_pretrained(
            checkpoint_path=PRETRAIN_DAILY,
            model_cfg=MODEL_CFG, n_features=n_features,
            level=level_name, level_cfg=level_cfg,
            cot_cfg={"embedding_dim": 384, "projection_dim": 128},
        )

        if level_name == "L3":
            dummy_loader = DataLoader(train_ds, batch_size=64, shuffle=False,
                                      collate_fn=custom_collate, num_workers=0)
            activations = model.get_expert_activations(dummy_loader, device)
            model.freeze_experts(activations["expert_ranks"])
            logger.info(f"  Frozen top-{model._freeze_n} experts: {activations['expert_ranks'][:model._freeze_n]}")

        model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR, weight_decay=0.01
        )
        total_steps = EPOCHS * len(train_loader)
        warmup = int(total_steps * 0.1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos',
        )

        loss_w = LOSS_WEIGHTS[level_name]
        n_pos = sum(1 for w in train_ds.windows if w["label"] == 1)
        n_neg = len(train_ds) - n_pos
        pos_ratio = n_pos / max(len(train_ds), 1)
        focal = FocalLoss(gamma=2.0, alpha=1.0 - pos_ratio).to(device)

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
                    cot_sim = torch.nn.functional.cosine_similarity(
                        out["cls_proj"], out["cot_target_proj"], dim=-1)
                    cot_loss = 1 - cot_sim
                    if has_cot is not None:
                        cot_loss = (cot_loss * has_cot).sum() / (has_cot.sum() + 1e-9)
                    else:
                        cot_loss = cot_loss.mean()
                    loss = loss + loss_w["cot_distill"] * cot_loss

                if "moe_loss" in out:
                    loss = loss + 0.001 * out["moe_loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sched.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            model.eval()
            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                    out = model(batch_gpu)
                    probs = torch.softmax(out["logits"], dim=-1)[:, 1]
                    preds = (probs > 0.5).long()
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(batch_gpu["label"].cpu().numpy())

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)
            m = compute_all_metrics(y_true, y_pred, y_prob)

            if (epoch + 1) % 10 == 0 or m["mcc"] > best_val_mcc:
                logger.info(f"  [MoE-{level_name}] F{fold_idx} Ep{epoch+1} "
                           f"MCC={m['mcc']:.4f} F1={m['f1']:.4f} AUC={m.get('roc_auc',0):.4f}")

            if m["mcc"] > best_val_mcc:
                best_val_mcc = m["mcc"]
                best_metrics = m.copy()
                patience_counter = 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "metrics": m}, fold_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"  Early stop at ep {epoch+1}")
                    break

        fold_metrics.append({"fold": fold_idx, "best_mcc": best_val_mcc, "metrics": best_metrics})
        logger.info(f"  Fold {fold_idx} best MCC: {best_val_mcc:.4f}")

    mccs = [fm["best_mcc"] for fm in fold_metrics]
    key = f"MoE-{level_name}"
    moe_results[key] = {
        "fold_metrics": fold_metrics,
        "mcc": float(np.mean(mccs)),
        "mcc_std": float(np.std(mccs)),
    }
    for metric_name in ["f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc"]:
        vals = [fm["metrics"].get(metric_name, 0) for fm in fold_metrics]
        moe_results[key][metric_name] = float(np.mean(vals))
        moe_results[key][f"{metric_name}_std"] = float(np.std(vals))

    logger.info(f"\nMoE-{level_name}: MCC={np.mean(mccs):.4f}±{np.std(mccs):.4f} "
                f"F1={moe_results[key]['f1']:.4f}")


# ==================================================================
# Part 2: Baselines (same 5-fold CV splits)
# ==================================================================
logger.info(f"\n{'='*60}")
logger.info("BASELINES")
logger.info(f"{'='*60}")

baseline_results = {}

for bname in ["RandomForest", "XGBoost", "SVM", "LSTM", "Transformer"]:
    logger.info(f"\n--- {bname} ---")
    fold_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        train_ds = AStockFinetuneDataset(
            cases=train_cases, astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings={},
        )
        val_ds = AStockFinetuneDataset(
            cases=val_cases, astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings={},
        )
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        n_features = train_ds.get_n_features()

        if bname in ["RandomForest", "XGBoost", "SVM"]:
            X_train, y_train = flatten_windows(train_ds)
            X_val, y_val = flatten_windows(val_ds)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            if bname == "RandomForest":
                clf = RandomForestClassifier(
                    n_estimators=300, max_depth=12, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1)
            elif bname == "XGBoost":
                scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                clf = XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    scale_pos_weight=min(scale, 5.0), subsample=0.8,
                    colsample_bytree=0.8, random_state=42, eval_metric="logloss",
                ) if HAS_XGB else None
                if clf is None:
                    continue
            else:
                clf = SVC(kernel="rbf", C=10, gamma="scale",
                          class_weight="balanced", probability=True, random_state=42)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            y_prob = clf.predict_proba(X_val)[:, 1]
            m = compute_all_metrics(y_val, y_pred, y_prob)

        else:
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=custom_collate, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    collate_fn=custom_collate, num_workers=2, pin_memory=True)

            if bname == "LSTM":
                bl_model = LSTMClassifier(n_features=n_features, d_hidden=128, n_layers=2, dropout=0.2)
            else:
                bl_model = TransformerClassifier(
                    n_features=n_features, d_model=256, n_heads=8,
                    n_layers=6, dropout=0.1, max_seq_len=512)

            bl_model = bl_model.to(device)
            bl_opt = torch.optim.AdamW(bl_model.parameters(), lr=1e-4, weight_decay=0.01)
            bl_focal = FocalLoss(gamma=2.0, alpha=0.5).to(device)
            best_mcc_bl, no_improve_bl, best_state_bl = -1.0, 0, None

            for ep in range(50):
                bl_model.train()
                for batch in train_loader:
                    bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
                    out = bl_model(bg)
                    loss = bl_focal(out["logits"], bg["label"])
                    bl_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(bl_model.parameters(), 1.0)
                    bl_opt.step()

                bl_model.eval()
                ap, al, apr = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch.items()}
                        out = bl_model(bg)
                        probs = torch.softmax(out["logits"], dim=-1)[:, 1]
                        ap.extend((probs > 0.5).long().cpu().numpy())
                        apr.extend(probs.cpu().numpy())
                        al.extend(bg["label"].cpu().numpy())

                ep_m = compute_all_metrics(np.array(al), np.array(ap), np.array(apr))
                if ep_m["mcc"] > best_mcc_bl + 0.001:
                    best_mcc_bl = ep_m["mcc"]
                    best_state_bl = {k: v.cpu().clone() for k, v in bl_model.state_dict().items()}
                    no_improve_bl = 0
                else:
                    no_improve_bl += 1
                    if no_improve_bl >= 10:
                        break

            if best_state_bl:
                bl_model.load_state_dict(best_state_bl)
            bl_model.to(device).eval()
            ap, al, apr = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
                    out = bl_model(bg)
                    probs = torch.softmax(out["logits"], dim=-1)[:, 1]
                    ap.extend((probs > 0.5).long().cpu().numpy())
                    apr.extend(probs.cpu().numpy())
                    al.extend(bg["label"].cpu().numpy())
            m = compute_all_metrics(np.array(al), np.array(ap), np.array(apr))

            del bl_model
            torch.cuda.empty_cache()

        fold_metrics.append({"fold": fold_idx, "best_mcc": m["mcc"], "metrics": m})
        logger.info(f"  {bname} Fold {fold_idx}: MCC={m['mcc']:.4f} F1={m['f1']:.4f}")

    mccs = [fm["best_mcc"] for fm in fold_metrics]
    baseline_results[bname] = {
        "fold_metrics": fold_metrics,
        "mcc": float(np.mean(mccs)) if mccs else 0,
        "mcc_std": float(np.std(mccs)) if mccs else 0,
    }
    for metric_name in ["f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc"]:
        vals = [fm["metrics"].get(metric_name, 0) for fm in fold_metrics]
        baseline_results[bname][metric_name] = float(np.mean(vals)) if vals else 0
        baseline_results[bname][f"{metric_name}_std"] = float(np.std(vals)) if vals else 0

    logger.info(f"{bname}: MCC={baseline_results[bname]['mcc']:.4f}±{baseline_results[bname]['mcc_std']:.4f}")


# ==================================================================
# Part 3: Save and summarize
# ==================================================================
all_results = {}
all_results.update(moe_results)
all_results.update(baseline_results)

with open(OUTPUT_DIR / "comprehensive_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

logger.info(f"\n{'='*60}")
logger.info("COMPREHENSIVE RESULTS")
logger.info(f"{'='*60}")
logger.info(f"{'Method':<25} {'MCC':>8} {'F1':>8} {'ROC-AUC':>8} {'PR-AUC':>8}")
logger.info("-" * 60)
ranked = sorted(all_results.items(), key=lambda x: x[1].get("mcc", 0), reverse=True)
for name, res in ranked:
    logger.info(f"{name:<25} {res['mcc']:>8.4f} {res.get('f1',0):>8.4f} "
                f"{res.get('roc_auc',0):>8.4f} {res.get('pr_auc',0):>8.4f}")
logger.info("=" * 60)
logger.info(f"Results saved to {OUTPUT_DIR / 'comprehensive_results.json'}")
