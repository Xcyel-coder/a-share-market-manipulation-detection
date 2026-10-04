"""
Baseline models for comparison experiments.
Traditional ML: RF, XGBoost, SVM
Deep Learning: LSTM, Transformer (no MoE)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from loguru import logger


# ============================================================
# Traditional ML Baselines
# ============================================================

def flatten_windows(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert windowed dataset to flat feature matrix for traditional ML."""
    features_list, labels_list = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        feat = sample["features"].numpy()  # (seq_len, n_feat)
        stats = np.concatenate([
            feat.mean(axis=0),
            feat.std(axis=0),
            feat.max(axis=0),
            feat.min(axis=0),
            feat[-1] - feat[0],  # trend
        ])
        features_list.append(stats)
        labels_list.append(sample["label"].item())
    return np.array(features_list), np.array(labels_list)


def run_traditional_ml(train_ds, val_ds) -> dict:
    """Run RF, XGBoost, SVM on flattened window features."""
    logger.info("Preparing traditional ML features...")
    X_train, y_train = flatten_windows(train_ds)
    X_val, y_val = flatten_windows(val_ds)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    results = {}

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42, eval_metric="logloss",
            scale_pos_weight=1.0, use_label_encoder=False,
        ) if HAS_XGBOOST else GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        ),
    }

    for name, clf in models.items():
        logger.info(f"Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)
        results[name] = {"y_pred": y_pred, "y_prob": y_prob, "y_true": y_val}
        logger.info(f"{name} done")

    return results


# ============================================================
# Deep Learning Baselines
# ============================================================

class LSTMClassifier(nn.Module):
    """Bidirectional LSTM baseline for time series classification."""

    def __init__(self, n_features, d_hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, d_hidden, n_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, batch):
        x = batch["features"]
        out, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        logits = self.classifier(h_cat)
        return {"logits": logits, "moe_loss": torch.tensor(0.0, device=x.device)}


class TransformerClassifier(nn.Module):
    """Vanilla Transformer encoder baseline (no MoE)."""

    def __init__(self, n_features, d_model=256, n_heads=8, n_layers=6, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, batch):
        x = batch["features"]
        B, S, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, :S + 1, :]
        x = self.encoder(x)
        x = self.norm(x)
        logits = self.classifier(x[:, 0])
        return {"logits": logits, "moe_loss": torch.tensor(0.0, device=x.device)}


def train_dl_baseline(model, train_loader, val_loader, device, name="model",
                       epochs=50, lr=1e-4, patience=10):
    """Train a DL baseline model with early stopping."""
    from .eval_framework import full_evaluation
    from training.finetune_trainer import FocalLoss

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    best_f1, no_improve = 0.0, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for batch in train_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch_gpu)
            loss = loss_fn(outputs["logits"], batch_gpu["label"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(batch_gpu)
                probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
                preds = (probs > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_gpu["label"].cpu().numpy())

        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        if f1 > best_f1 + 0.001:
            best_f1 = f1
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch_gpu)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_gpu["label"].cpu().numpy())

    return {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_prob": np.array(all_probs),
    }
