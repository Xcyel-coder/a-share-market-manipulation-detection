"""
Comprehensive evaluation framework for Phase 4.
Covers classification, ranking, interpretability, and timeliness metrics.
"""
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, matthews_corrcoef,
)
from loguru import logger


def evaluate_classification(y_true, y_pred, y_prob):
    """Full classification metrics suite."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        "specificity": tn / (tn + fp + 1e-9),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    return metrics


def evaluate_ranking(y_true, y_prob):
    """Ranking metrics: ROC-AUC and PR-AUC."""
    metrics = {}
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = 0.5
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["pr_auc"] = 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    metrics["pr_curve"] = {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()}
    return metrics


def evaluate_timeliness(cases, predictions, case_window_map):
    """
    Compute warning lead days: how many days before publish_date
    the model first flags a case as manipulation.
    
    case_window_map: dict mapping case doc_id -> list of (window_idx, date, pred, prob)
    """
    lead_days = []
    for case in cases:
        doc_id = case.get("doc_id", "")
        publish_date_str = case.get("publish_date", "")
        if not publish_date_str or doc_id not in case_window_map:
            continue

        try:
            pub_date = datetime.strptime(publish_date_str[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        windows = case_window_map[doc_id]
        first_warning = None
        for _, date_str, pred, prob in sorted(windows, key=lambda x: x[1]):
            if pred == 1:
                try:
                    warn_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                    if first_warning is None or warn_date < first_warning:
                        first_warning = warn_date
                except (ValueError, TypeError):
                    continue

        if first_warning is not None:
            delta = (pub_date - first_warning).days
            lead_days.append({"doc_id": doc_id, "lead_days": delta})

    if not lead_days:
        return {"mean_lead_days": 0, "median_lead_days": 0, "details": []}

    days = [d["lead_days"] for d in lead_days]
    return {
        "mean_lead_days": float(np.mean(days)),
        "median_lead_days": float(np.median(days)),
        "min_lead_days": float(np.min(days)),
        "max_lead_days": float(np.max(days)),
        "n_cases_with_warning": len(lead_days),
        "details": lead_days,
    }


def evaluate_interpretability(cot_results, cases):
    """
    Evaluate CoT interpretability quality.
    - Average quality score from teacher model
    - Keyword coverage (manipulation method terms found in CoT text)
    """
    manipulation_keywords = [
        "连续交易", "对倒", "虚假申报", "拉抬", "操纵", "异常",
        "大额买入", "集中交易", "尾市", "涨停", "成交量",
        "资金优势", "持股优势", "信息优势", "拉升", "打压",
    ]

    quality_scores = []
    keyword_coverage = []
    method_match_count = 0
    total_with_methods = 0

    case_methods = {}
    for c in cases:
        methods = c.get("manipulation_methods", [])
        if methods:
            case_methods[c.get("doc_id", "")] = methods

    for entry in cot_results:
        q = entry.get("quality_score", 0)
        if q > 0:
            quality_scores.append(q)

        text = entry.get("cot_text", "")
        if not text:
            continue

        found = sum(1 for kw in manipulation_keywords if kw in text)
        keyword_coverage.append(found / len(manipulation_keywords))

        doc_id = entry.get("doc_id", "")
        if doc_id in case_methods:
            total_with_methods += 1
            methods = case_methods[doc_id]
            if any(m in text for m in methods):
                method_match_count += 1

    return {
        "avg_quality_score": float(np.mean(quality_scores)) if quality_scores else 0,
        "avg_keyword_coverage": float(np.mean(keyword_coverage)) if keyword_coverage else 0,
        "method_match_rate": method_match_count / max(total_with_methods, 1),
        "total_cot_entries": len(cot_results),
        "entries_with_quality": len(quality_scores),
    }


def full_evaluation(y_true, y_pred, y_prob, name="model"):
    """Run all metric suites and return combined dict."""
    cls_metrics = evaluate_classification(y_true, y_pred, y_prob)
    rank_metrics = evaluate_ranking(y_true, y_prob)
    combined = {**cls_metrics, **{k: v for k, v in rank_metrics.items() if k not in ("roc_curve", "pr_curve")}}
    combined["curves"] = {
        "roc": rank_metrics.get("roc_curve", {}),
        "pr": rank_metrics.get("pr_curve", {}),
    }

    logger.info(f"[{name}] F1={combined['f1']:.4f} P={combined['precision']:.4f} "
                f"R={combined['recall']:.4f} ROC-AUC={combined['roc_auc']:.4f} "
                f"PR-AUC={combined['pr_auc']:.4f} MCC={combined['mcc']:.4f}")
    return combined
