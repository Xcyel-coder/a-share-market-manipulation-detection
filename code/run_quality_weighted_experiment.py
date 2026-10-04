#!/usr/bin/env python3
"""
Quality-Weighted Comprehensive Experiment.

Key changes from run_comprehensive_experiment.py:
  1. Cases loaded from manipulation_cases_verified.csv (with quality tiers)
  2. Confidence-weighted FocalLoss: Gold=1.0, Silver=0.6, Bronze=0.3
  3. Stratified CV: Gold cases preferentially in validation for reliable metrics
  4. MoE L0-L3 + all baselines, same architecture
"""
import sys, json, os, time, csv, torch, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/project/model")
os.chdir("/root/project/model")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/root/project/outputs/quality_weighted/experiment.log", rotation="50MB", level="DEBUG")

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

VERIFIED_CSV = "/root/project/data/manipulation_cases_verified.csv"
CASES_DIR = "/root/project/data/processed/extracted_cases"
ASTOCK_DIR = "/root/project/data/raw/astock"
COT_NPZ = "/root/project/model/outputs/cot/cot_embeddings.npz"
PRETRAIN_DAILY = "/root/project/outputs/pretrain_daily/align_daily/best.pt"
OUTPUT_DIR = Path("/root/project/outputs/quality_weighted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CFG = {
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001},
}
FEAT_CFG = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}

QUALITY_WEIGHTS = {"Gold": 1.0, "Silver": 0.6, "Bronze": 0.3}

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


class QualityWeightedFocalLoss(torch.nn.Module):
    """FocalLoss with per-sample quality confidence weights."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, quality_weights=None):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if quality_weights is not None:
            focal_loss = focal_loss * quality_weights

        return focal_loss.mean()


# ==================================================================
# Load cases from verified CSV
# ==================================================================
logger.info("Loading verified cases from CSV...")

csv_cases = []
with open(VERIFIED_CSV, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_cases.append(row)

logger.info(f"CSV records: {len(csv_cases)}")

# Build lookup from existing extracted JSONs
extracted_lookup = {}
for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
        doc_id = c.get("doc_id", "")
        extracted_lookup[doc_id] = c

logger.info(f"Extracted JSON lookup: {len(extracted_lookup)} entries")

# Match CSV to extracted cases and build final case list
cases = []
quality_map = {}  # doc_id -> quality_weight

for row in csv_cases:
    stock_code = row.get("stock_code", "")
    v_start = row.get("verified_start_date", "")
    v_end = row.get("verified_end_date", "")
    tier = row.get("quality_tier", "Bronze")
    csrc_doc_id = row.get("csrc_doc_id", "")
    source = row.get("source", "")

    if not v_start or not v_end or not stock_code:
        continue

    # Try to find matching extracted JSON by doc_id
    matched_case = None
    for doc_id, ext in extracted_lookup.items():
        ext_code = ext.get("stock_code", "")
        if ext_code and ext_code[:6] == stock_code[:6]:
            astock_path = Path(ASTOCK_DIR) / doc_id / "daily.parquet"
            if astock_path.exists():
                matched_case = ext.copy()
                matched_case["manipulation_start_date"] = v_start
                matched_case["manipulation_end_date"] = v_end
                matched_case["quality_tier"] = tier
                break

    if not matched_case:
        # Try CSRC doc_id directly
        if csrc_doc_id and csrc_doc_id in extracted_lookup:
            ext = extracted_lookup[csrc_doc_id]
            astock_path = Path(ASTOCK_DIR) / csrc_doc_id / "daily.parquet"
            if astock_path.exists():
                matched_case = ext.copy()
                matched_case["manipulation_start_date"] = v_start
                matched_case["manipulation_end_date"] = v_end
                matched_case["quality_tier"] = tier

    if matched_case:
        doc_id = matched_case["doc_id"]
        if doc_id not in quality_map:
            cases.append(matched_case)
            quality_map[doc_id] = QUALITY_WEIGHTS.get(tier, 0.3)

n_gold = sum(1 for c in cases if c.get("quality_tier") == "Gold")
n_silver = sum(1 for c in cases if c.get("quality_tier") == "Silver")
n_bronze = sum(1 for c in cases if c.get("quality_tier") == "Bronze")
logger.info(f"Matched cases: {len(cases)} (Gold={n_gold}, Silver={n_silver}, Bronze={n_bronze})")

cot_embeddings = {}
if Path(COT_NPZ).exists():
    npz_data = np.load(COT_NPZ)
    cot_embeddings = {k: npz_data[k] for k in npz_data.files}
    logger.info(f"CoT embeddings: {len(cot_embeddings)} cases")


# ==================================================================
# Quality-aware CV splits: ensure Gold cases are well-represented
# ==================================================================
def build_quality_cv_splits(cases, n_folds=5, seed=42):
    """Stratified splits ensuring each fold's validation has Gold cases."""
    rng = np.random.RandomState(seed)

    gold = [i for i, c in enumerate(cases) if c.get("quality_tier") == "Gold"]
    others = [i for i, c in enumerate(cases) if c.get("quality_tier") != "Gold"]
    rng.shuffle(gold)
    rng.shuffle(others)

    splits = []
    gold_fold_size = max(len(gold) // n_folds, 1)
    other_fold_size = max(len(others) // n_folds, 1)

    for fold in range(n_folds):
        g_start = fold * gold_fold_size
        g_end = g_start + gold_fold_size if fold < n_folds - 1 else len(gold)
        o_start = fold * other_fold_size
        o_end = o_start + other_fold_size if fold < n_folds - 1 else len(others)

        val_indices = set(gold[g_start:g_end]) | set(others[o_start:o_end])
        train_cases = [cases[i] for i in range(len(cases)) if i not in val_indices]
        val_cases = [cases[i] for i in val_indices]
        splits.append((train_cases, val_cases))

    return splits


splits = build_quality_cv_splits(cases, n_folds=5, seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}, Folds: {len(splits)}")
for i, (tr, va) in enumerate(splits):
    va_gold = sum(1 for c in va if c.get("quality_tier") == "Gold")
    logger.info(f"  Fold {i}: train={len(tr)}, val={len(va)} (Gold in val: {va_gold})")


# ==================================================================
# Custom dataset wrapper that carries quality_weight per window
# ==================================================================
class QualityAwareDataset(AStockFinetuneDataset):
    """Extends AStockFinetuneDataset to carry quality_weight per window."""

    def __init__(self, cases, quality_map, **kwargs):
        super().__init__(cases=cases, **kwargs)
        self._quality_map = quality_map
        for w in self.windows:
            w["quality_weight"] = self._quality_map.get(w["doc_id"], 0.3)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["quality_weight"] = torch.tensor(
            self.windows[idx]["quality_weight"], dtype=torch.float32)
        return item


# ==================================================================
# Part 1: MoE L0-L3 with quality-weighted loss
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
    logger.info(f"MoE-{level_name} (quality-weighted, daily-aligned pretrain)")
    logger.info(f"{'='*60}")

    level_dir = OUTPUT_DIR / f"MoE_{level_name}"
    level_dir.mkdir(exist_ok=True)
    fold_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx} ({len(train_cases)} train, {len(val_cases)} val) ---")
        fold_dir = level_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        train_ds = QualityAwareDataset(
            cases=train_cases, quality_map=quality_map,
            astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings=cot_embeddings,
        )
        val_ds = QualityAwareDataset(
            cases=val_cases, quality_map=quality_map,
            astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings=cot_embeddings,
        )
        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning(f"  Empty dataset, skip fold {fold_idx}")
            continue

        n_features = train_ds.get_n_features()

        labels = [w["label"] for w in train_ds.windows]
        class_counts = np.bincount(labels)
        sample_weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)

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
            model.to(device)
            dummy_loader = DataLoader(train_ds, batch_size=64, shuffle=False,
                                      collate_fn=custom_collate, num_workers=0)
            activations = model.get_expert_activations(dummy_loader, device)
            model.freeze_experts(activations["expert_ranks"])
            logger.info(f"  Frozen top-{model._freeze_n} experts: {activations['expert_ranks'][:model._freeze_n]}")
        else:
            model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR, weight_decay=0.01
        )
        total_steps = EPOCHS * len(train_loader)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos',
        )

        loss_w = LOSS_WEIGHTS[level_name]
        n_pos = sum(1 for w in train_ds.windows if w["label"] == 1)
        pos_ratio = n_pos / max(len(train_ds), 1)
        focal = QualityWeightedFocalLoss(gamma=2.0, alpha=1.0 - pos_ratio).to(device)

        best_val_mcc = -1.0
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                labels_t = batch_gpu["label"]
                qw = batch_gpu.get("quality_weight")
                out = model(batch_gpu)

                loss = loss_w["classification"] * focal(out["logits"], labels_t, qw)

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
                            "metrics": m, "level": level_name, "fold": fold_idx},
                           fold_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"  Early stop at ep {epoch+1}")
                    break

        fold_metrics.append({"fold": fold_idx, "best_mcc": best_val_mcc, "metrics": best_metrics})
        logger.info(f"  Fold {fold_idx} best MCC: {best_val_mcc:.4f}")

        del model
        torch.cuda.empty_cache()

    mccs = [fm["best_mcc"] for fm in fold_metrics]
    key = f"MoE-{level_name}"
    moe_results[key] = {
        "fold_metrics": fold_metrics,
        "mcc": float(np.mean(mccs)),
        "mcc_std": float(np.std(mccs)),
    }
    for metric_name in ["f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc", "specificity"]:
        vals = [fm["metrics"].get(metric_name, 0) for fm in fold_metrics]
        moe_results[key][metric_name] = float(np.mean(vals))
        moe_results[key][f"{metric_name}_std"] = float(np.std(vals))

    logger.info(f"\nMoE-{level_name}: MCC={np.mean(mccs):.4f}+/-{np.std(mccs):.4f} "
                f"F1={moe_results[key]['f1']:.4f}")


# ==================================================================
# Part 2: Baselines (same quality-aware CV splits)
# ==================================================================
logger.info(f"\n{'='*60}")
logger.info("BASELINES")
logger.info(f"{'='*60}")

baseline_results = {}

for bname in ["RandomForest", "XGBoost", "SVM", "LSTM", "Transformer"]:
    logger.info(f"\n--- {bname} ---")
    fold_metrics = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        train_ds = QualityAwareDataset(
            cases=train_cases, quality_map=quality_map,
            astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings={},
        )
        val_ds = QualityAwareDataset(
            cases=val_cases, quality_map=quality_map,
            astock_dir=ASTOCK_DIR, feature_cfg=FEAT_CFG,
            seq_len=60, stride=5, min_bars=30, overlap_ratio=0.5,
            oversample_positive=1, cot_embeddings={},
        )
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        n_features = train_ds.get_n_features()

        if bname in ["RandomForest", "XGBoost", "SVM"]:
            X_train, y_train = flatten_windows(train_ds)
            X_val, y_val = flatten_windows(val_ds)

            # Quality-weighted sample weights for sklearn
            train_qw = np.array([w["quality_weight"] for w in train_ds.windows])

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            if bname == "RandomForest":
                clf = RandomForestClassifier(
                    n_estimators=300, max_depth=12, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1)
                clf.fit(X_train, y_train, sample_weight=train_qw)
            elif bname == "XGBoost":
                scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                clf = XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    scale_pos_weight=min(scale, 5.0), subsample=0.8,
                    colsample_bytree=0.8, random_state=42, eval_metric="logloss",
                ) if HAS_XGB else None
                if clf is None:
                    continue
                clf.fit(X_train, y_train, sample_weight=train_qw)
            else:
                clf = SVC(kernel="rbf", C=10, gamma="scale",
                          class_weight="balanced", probability=True, random_state=42)
                clf.fit(X_train, y_train, sample_weight=train_qw)

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
            bl_focal = QualityWeightedFocalLoss(gamma=2.0, alpha=0.5).to(device)
            best_mcc_bl, no_improve_bl, best_state_bl = -1.0, 0, None

            for ep in range(50):
                bl_model.train()
                for batch in train_loader:
                    bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
                    qw = bg.get("quality_weight")
                    out = bl_model(bg)
                    loss = bl_focal(out["logits"], bg["label"], qw)
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
    for metric_name in ["f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc", "specificity"]:
        vals = [fm["metrics"].get(metric_name, 0) for fm in fold_metrics]
        baseline_results[bname][metric_name] = float(np.mean(vals)) if vals else 0
        baseline_results[bname][f"{metric_name}_std"] = float(np.std(vals)) if vals else 0

    logger.info(f"{bname}: MCC={baseline_results[bname]['mcc']:.4f}+/-{baseline_results[bname]['mcc_std']:.4f}")


# ==================================================================
# Part 3: Save all results
# ==================================================================
all_results = {
    "experiment": "quality_weighted",
    "quality_weights": QUALITY_WEIGHTS,
    "n_cases": len(cases),
    "n_gold": n_gold,
    "n_silver": n_silver,
    "n_bronze": n_bronze,
}
all_results.update(moe_results)
all_results.update(baseline_results)

with open(OUTPUT_DIR / "quality_weighted_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

logger.info(f"\n{'='*60}")
logger.info("QUALITY-WEIGHTED RESULTS")
logger.info(f"{'='*60}")
logger.info(f"Cases: {len(cases)} (Gold={n_gold}, Silver={n_silver}, Bronze={n_bronze})")
logger.info(f"Weights: {QUALITY_WEIGHTS}")
logger.info(f"{'Method':<25} {'MCC':>8} {'F1':>8} {'ROC-AUC':>8} {'PR-AUC':>8}")
logger.info("-" * 60)
ranked = sorted(
    [(k, v) for k, v in all_results.items() if isinstance(v, dict) and "mcc" in v],
    key=lambda x: x[1].get("mcc", 0), reverse=True
)
for name, res in ranked:
    logger.info(f"{name:<25} {res['mcc']:>8.4f} {res.get('f1',0):>8.4f} "
                f"{res.get('roc_auc',0):>8.4f} {res.get('pr_auc',0):>8.4f}")
logger.info("=" * 60)
logger.info(f"Results saved to {OUTPUT_DIR / 'quality_weighted_results.json'}")
