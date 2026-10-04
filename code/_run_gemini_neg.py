"""Supplement Gemini zero-shot with negative samples (pre-manipulation periods)."""
import os
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CASES_DIR = Path(__file__).parent / "data_pipeline" / "data" / "processed" / "extracted_cases"
ASTOCK_DIR = Path(__file__).parent / "data_pipeline" / "data" / "raw" / "astock"
OUTPUT_DIR = Path(__file__).parent / "model" / "outputs" / "phase4"

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


def call_gemini(prompt, max_retries=3):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                print(f"  Error {resp.status_code}")
                time.sleep(3)
        except Exception as e:
            print(f"  {e}")
            time.sleep(3)
    return None


def parse_response(text):
    try:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        result = json.loads(text)
        is_manip = result.get("judgment", "").startswith("是")
        conf = float(result.get("confidence", 50)) / 100.0
        return is_manip, conf, result.get("reason", "")
    except Exception:
        return "是" in text[:30], 0.5, text[:100]


def get_window_stats(df, end_idx, window=60):
    start_idx = max(0, end_idx - window)
    w = df.iloc[start_idx:end_idx]
    if len(w) < 10:
        return None
    close = w["close"].values
    volume = w["volume"].values
    returns = np.diff(close) / (close[:-1] + 1e-9)
    vol_ratio = volume / (volume.mean() + 1e-9)
    return {
        "period": f"{w['date'].iloc[0].strftime('%Y-%m-%d')} to {w['date'].iloc[-1].strftime('%Y-%m-%d')}",
        "trading_days": len(w),
        "price_change_pct": f"{((close[-1]/close[0])-1)*100:.1f}%",
        "max_daily_return": f"{returns.max()*100:.1f}%",
        "min_daily_return": f"{returns.min()*100:.1f}%",
        "avg_daily_return": f"{returns.mean()*100:.2f}%",
        "return_volatility": f"{returns.std()*100:.2f}%",
        "max_volume_ratio": f"{vol_ratio.max():.1f}x",
        "price_range": f"{close.min():.2f} - {close.max():.2f}",
    }


def build_prompt(stock_code, stock_name, feat_during, feat_before):
    return f"""你是一位中国A股市场监管分析专家。请根据以下股票交易数据特征，判断该股票是否存在市场操纵行为。

## 股票信息
- 股票代码: {stock_code}
- 股票名称: {stock_name}

## 待分析期间的交易特征
- 分析期间: {feat_during['period']}
- 交易天数: {feat_during['trading_days']}
- 期间涨跌幅: {feat_during['price_change_pct']}
- 最大单日涨幅: {feat_during['max_daily_return']}
- 最大单日跌幅: {feat_during['min_daily_return']}
- 收益率波动率: {feat_during['return_volatility']}
- 最大成交量比率: {feat_during['max_volume_ratio']}
- 价格区间: {feat_during['price_range']}

## 前期对照
- 对照期间: {feat_before['period']}
- 期间涨跌幅: {feat_before['price_change_pct']}
- 收益率波动率: {feat_before['return_volatility']}
- 最大成交量比率: {feat_before['max_volume_ratio']}

请严格按以下JSON格式回答:
{{"judgment": "是/否", "confidence": 0-100, "reason": "一句话理由"}}"""


# Load existing results
with open(OUTPUT_DIR / "gemini_zeroshot_results.json", encoding="utf-8") as f:
    existing = json.load(f)

details = existing["details"]
y_true = [d["true_label"] for d in details]
y_pred = [d["pred_label"] for d in details]
y_prob = [d["confidence"] for d in details]

print(f"Existing results: {len(details)} positive cases")

# Generate negative samples from early stock data
cases = []
for f in sorted(CASES_DIR.glob("*_extracted.json")):
    with open(f, "r", encoding="utf-8") as fh:
        case = json.load(fh)
        if case.get("stock_code") and case.get("manipulation_start_date"):
            cases.append(case)

neg_tested = 0
for case in cases:
    doc_id = case["doc_id"]
    stock_code = case["stock_code"]
    manip_start = case["manipulation_start_date"]

    parquet_path = ASTOCK_DIR / doc_id / "daily.parquet"
    if not parquet_path.exists():
        continue

    try:
        df = pd.read_parquet(parquet_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        manip_start_dt = pd.to_datetime(manip_start)
        pre_manip = df[df["date"] < manip_start_dt]

        if len(pre_manip) < 130:
            continue

        feat_during = get_window_stats(pre_manip, len(pre_manip) - 60, 60)
        feat_before = get_window_stats(pre_manip, len(pre_manip) - 120, 60)

        if not feat_during or not feat_before:
            continue

        prompt = build_prompt(stock_code, case.get("stock_name", ""), feat_during, feat_before)
        text = call_gemini(prompt)
        if not text:
            continue

        is_manip, conf, reason = parse_response(text)

        y_true.append(0)
        y_pred.append(1 if is_manip else 0)
        y_prob.append(conf)

        details.append({
            "doc_id": doc_id, "stock_code": stock_code,
            "true_label": 0, "pred_label": 1 if is_manip else 0,
            "confidence": conf, "reason": reason,
            "window_type": "pre-manipulation (negative)",
        })

        neg_tested += 1
        label = "操纵(FP)" if is_manip else "正常(TN)"
        print(f"  [{doc_id} neg] -> {label} (conf={conf:.0%})")
        time.sleep(1)

    except Exception as e:
        print(f"  Error for {doc_id}: {e}")

print(f"\nNegative samples tested: {neg_tested}")

# Recompute metrics
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, accuracy_score

y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
    "n_positive": int(y_true.sum()),
    "n_negative": int((1 - y_true).sum()),
    "n_total": len(y_true),
}
try:
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
except ValueError:
    metrics["roc_auc"] = 0.5

print(f"\n{'='*60}")
print(f"Gemini Zero-Shot (with negatives):")
print(f"  F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}")
print(f"  ROC-AUC={metrics['roc_auc']:.4f} MCC={metrics['mcc']:.4f}")
print(f"  Samples: {metrics['n_positive']} pos, {metrics['n_negative']} neg")
print(f"{'='*60}")

output = {"metrics": metrics, "details": details}
with open(OUTPUT_DIR / "gemini_zeroshot_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

comp_path = OUTPUT_DIR / "comparison_results.json"
with open(comp_path) as f:
    comp = json.load(f)
comp["Gemini-ZeroShot"] = metrics
with open(comp_path, "w") as f:
    json.dump(comp, f, indent=2, default=str)

print("Updated all results.")
