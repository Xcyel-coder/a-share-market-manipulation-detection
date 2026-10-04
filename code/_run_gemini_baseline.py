"""
Phase 4 补充：商业大模型零样本推理基线
使用 Gemini 2.5 Flash 对 A 股案例做零样本操纵判断
"""
import os
import json
import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CASES_DIR = Path(__file__).parent / "data_pipeline" / "data" / "processed" / "extracted_cases"
ASTOCK_DIR = Path(__file__).parent / "data_pipeline" / "data" / "raw" / "astock"
OUTPUT_DIR = Path(__file__).parent / "model" / "outputs" / "phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import requests
import pandas as pd

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


def call_gemini(prompt, max_retries=3):
    """Call Gemini via REST API."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text
            elif resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(3)
        except Exception as e:
            print(f"  Request error: {e}")
            time.sleep(3)
    return None


def load_cases():
    cases = []
    for f in sorted(CASES_DIR.glob("*_extracted.json")):
        with open(f, "r", encoding="utf-8") as fh:
            case = json.load(fh)
            if case.get("stock_code") and case.get("manipulation_start_date"):
                cases.append(case)
    return cases


def get_stock_features(doc_id, date_str, window_days=60):
    """Extract key statistical features from stock data for a given window."""
    parquet_path = ASTOCK_DIR / doc_id / "daily.parquet"
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        target_date = pd.to_datetime(date_str)
        mask = df["date"] <= target_date
        window = df[mask].tail(window_days)

        if len(window) < 10:
            return None

        close = window["close"].values
        volume = window["volume"].values

        returns = np.diff(close) / (close[:-1] + 1e-9)
        vol_ratio = volume / (volume.mean() + 1e-9)

        features = {
            "period": f"{window['date'].iloc[0].strftime('%Y-%m-%d')} to {window['date'].iloc[-1].strftime('%Y-%m-%d')}",
            "trading_days": len(window),
            "price_change_pct": f"{((close[-1]/close[0])-1)*100:.1f}%",
            "max_daily_return": f"{returns.max()*100:.1f}%",
            "min_daily_return": f"{returns.min()*100:.1f}%",
            "avg_daily_return": f"{returns.mean()*100:.2f}%",
            "return_volatility": f"{returns.std()*100:.2f}%",
            "max_volume_ratio": f"{vol_ratio.max():.1f}x",
            "avg_volume_ratio": f"{vol_ratio.mean():.1f}x",
            "price_range": f"{close.min():.2f} - {close.max():.2f}",
        }

        if "turnover_rate" in window.columns:
            features["avg_turnover"] = f"{window['turnover_rate'].mean():.2f}%"

        return features
    except Exception as e:
        return None


def build_prompt(case, features_before, features_during):
    """Build zero-shot prompt for manipulation detection."""
    prompt = f"""你是一位中国A股市场监管分析专家。请根据以下股票交易数据特征，判断该股票是否存在市场操纵行为。

## 股票信息
- 股票代码: {case['stock_code']}
- 股票名称: {case.get('stock_name', '未知')}

## 待分析期间的交易特征
- 分析期间: {features_during['period']}
- 交易天数: {features_during['trading_days']}
- 期间涨跌幅: {features_during['price_change_pct']}
- 最大单日涨幅: {features_during['max_daily_return']}
- 最大单日跌幅: {features_during['min_daily_return']}
- 日均收益率: {features_during['avg_daily_return']}
- 收益率波动率: {features_during['return_volatility']}
- 最大成交量比率(相对均值): {features_during['max_volume_ratio']}
- 价格区间: {features_during['price_range']}

## 前期对照（操纵前60交易日）
- 对照期间: {features_before['period']}
- 期间涨跌幅: {features_before['price_change_pct']}
- 收益率波动率: {features_before['return_volatility']}
- 最大成交量比率: {features_before['max_volume_ratio']}

## 请回答
1. 判断: 该股票在分析期间是否存在操纵嫌疑? (是/否)
2. 置信度: 0-100之间的数字
3. 简要理由: 一句话说明

请严格按以下JSON格式回答，不要添加任何其他内容:
{{"judgment": "是/否", "confidence": 0-100, "reason": "一句话理由"}}"""
    return prompt


def parse_response(text):
    """Parse Gemini response."""
    try:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        is_manipulation = result.get("judgment", "").startswith("是")
        confidence = float(result.get("confidence", 50)) / 100.0
        reason = result.get("reason", "")
        return is_manipulation, confidence, reason
    except Exception:
        if "是" in text[:20]:
            return True, 0.6, text[:100]
        return False, 0.4, text[:100]


def run_zero_shot_baseline():
    cases = load_cases()
    print(f"Loaded {len(cases)} cases")

    results = []
    y_true, y_pred, y_prob = [], [], []

    for i, case in enumerate(cases):
        doc_id = case["doc_id"]
        stock_code = case["stock_code"]
        manip_start = case["manipulation_start_date"]
        manip_end = case.get("manipulation_end_date", manip_start)

        print(f"\n[{i+1}/{len(cases)}] {doc_id} ({stock_code})")

        features_during = get_stock_features(doc_id, manip_end)
        features_before = get_stock_features(doc_id, manip_start)

        if not features_during or not features_before:
            print(f"  Skipped: insufficient data")
            continue

        prompt = build_prompt(case, features_before, features_during)

        try:
            text = call_gemini(prompt)
            if not text:
                print(f"  Skipped: no API response")
                continue
            is_manip, conf, reason = parse_response(text)

            result = {
                "doc_id": doc_id,
                "stock_code": stock_code,
                "stock_name": case.get("stock_name", ""),
                "true_label": 1,
                "pred_label": 1 if is_manip else 0,
                "confidence": conf,
                "reason": reason,
            }
            results.append(result)

            y_true.append(1)
            y_pred.append(1 if is_manip else 0)
            y_prob.append(conf)

            print(f"  Judgment: {'操纵' if is_manip else '正常'} (conf={conf:.0%})")
            print(f"  Reason: {reason[:80]}")

        except Exception as e:
            print(f"  API Error: {e}")
            time.sleep(5)
            continue

        time.sleep(1)

    # Also test some negative windows (before manipulation period)
    print("\n=== Testing negative (pre-manipulation) windows ===")
    neg_count = 0
    for case in cases[:min(15, len(cases))]:
        doc_id = case["doc_id"]
        manip_start = case["manipulation_start_date"]

        try:
            start_dt = datetime.strptime(manip_start, "%Y-%m-%d")
            pre_date = (start_dt - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        except Exception:
            continue

        features_pre = get_stock_features(doc_id, pre_date)
        pre_before_date = (datetime.strptime(pre_date, "%Y-%m-%d") - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        features_before = get_stock_features(doc_id, pre_before_date)

        if not features_pre or not features_before:
            continue

        prompt_neg = build_prompt(
            {"stock_code": case["stock_code"], "stock_name": case.get("stock_name", "")},
            features_before, features_pre,
        )

        try:
            text_neg = call_gemini(prompt_neg)
            if not text_neg:
                continue
            is_manip, conf, reason = parse_response(text_neg)

            y_true.append(0)
            y_pred.append(1 if is_manip else 0)
            y_prob.append(conf)

            results.append({
                "doc_id": doc_id, "stock_code": case["stock_code"],
                "true_label": 0, "pred_label": 1 if is_manip else 0,
                "confidence": conf, "reason": reason,
                "window_type": "pre-manipulation",
            })

            neg_count += 1
            print(f"  [{doc_id} pre-manip] -> {'操纵' if is_manip else '正常'} (conf={conf:.0%})")
        except Exception as e:
            print(f"  API Error: {e}")
        time.sleep(1)

    print(f"\nNegative samples tested: {neg_count}")

    # Compute metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
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
    print(f"Gemini Zero-Shot Results:")
    print(f"  F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}")
    print(f"  ROC-AUC={metrics['roc_auc']:.4f} MCC={metrics['mcc']:.4f}")
    print(f"  Samples: {metrics['n_positive']} pos, {metrics['n_negative']} neg")
    print(f"{'='*60}")

    output = {"metrics": metrics, "details": results}
    with open(OUTPUT_DIR / "gemini_zeroshot_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # Update comparison results
    comp_path = OUTPUT_DIR / "comparison_results.json"
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        comp["Gemini-ZeroShot"] = metrics
        with open(comp_path, "w") as f:
            json.dump(comp, f, indent=2, default=str)
        print("Updated comparison_results.json with Gemini baseline.")

    return metrics


if __name__ == "__main__":
    run_zero_shot_baseline()
