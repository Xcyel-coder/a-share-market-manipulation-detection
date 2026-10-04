"""
Teacher model Chain-of-Thought (CoT) generation using Gemini 2.5 Flash.

For each A-share manipulation case, generates structured reasoning chains
that the student MoE model will learn to replicate in Phase 3.

Each case produces 3-4 CoT entries from different analysis perspectives:
  1. Primary detection reasoning (full analysis)
  2. Counter-factual reasoning (if this were normal trading)
  3. Feature-focused reasoning (which indicators matter most)
  4. Cross-market perspective (crypto parallels)
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger


COT_PROMPTS = {
    "primary": """你是一位资深的中国A股市场操纵检测专家。请针对以下证监会处罚案例，生成详细的逐步推理分析（Chain-of-Thought）。

## 案例信息
- 股票：{stock_name}（{stock_code}）
- 操纵期间：{start_date} 至 {end_date}（共 {duration} 天）
- 操纵手法：{methods}
- 案情描述：{description}
- 处罚金额：{fine_info}

## 请严格按以下结构输出你的推理过程：

### 1. 模式识别
本案涉及哪些类型的市场操纵？描述每种操纵手法的具体交易特征。

### 2. 技术指标异常分析
在操纵期间，以下指标会出现什么异常？
- 成交量（volume）
- 价格走势（price action）
- 委托流（order flow）
- 波动率（volatility）
- 换手率（turnover rate）

### 3. 自动化检测推理
假设你只有公开交易数据（K线、成交量、盘口数据），无内幕信息。逐步推理如何检测：
步骤1→ 步骤2→ 步骤3→ ...

### 4. 关键特征排序
列出最具区分性的5个量化特征，按重要性从高到低排序，并说明理由。

### 5. 检测难度评估
- 难度等级：[简单/中等/困难/极困难]
- 原因分析
- 与常见量化交易策略的区别

### 6. 结论
用2-3句话总结核心检测信号。""",

    "counterfactual": """你是一位A股市场操纵检测专家。以下是一个被证监会认定的市场操纵案例。请从"反事实"角度分析：假设这些交易行为是合法的正常交易，会有什么不同？

## 案例信息
- 股票：{stock_name}（{stock_code}）
- 操纵期间：{start_date} 至 {end_date}
- 操纵手法：{methods}
- 案情描述：{description}

## 请回答：

### 1. 正常交易 vs. 操纵交易
如果这些交易是正常行为，交易数据（成交量、价格、委托）会呈现什么不同的模式？

### 2. 关键区分信号
列出3-5个能区分"正常大额交易"和"操纵性交易"的量化指标。

### 3. 误判风险
在什么情况下，正常交易行为可能被错误地标记为操纵？如何降低误报率？

### 4. 结论
总结区分操纵与正常交易的核心逻辑。""",

    "feature_analysis": """你是一位金融量化分析师，专注于市场异常检测。分析以下操纵案例中最重要的量化特征。

## 案例信息
- 股票：{stock_name}（{stock_code}）
- 操纵手法：{methods}
- 操纵时长：{duration} 天
- 案情描述：{description}

## 请输出：

### 1. 时序特征设计
针对此类操纵，设计5个基于OHLCV数据的时序特征（给出计算公式）。

### 2. 统计异常阈值
每个特征的正常范围是什么？超过什么阈值应触发警报？

### 3. 多指标联合判断
如何组合多个指标来提高检测精度？描述一个多指标联合判断规则。

### 4. 时间尺度分析
这种操纵在1分钟、5分钟、日线级别分别呈现什么特征？哪个时间尺度最容易检测？""",

    "cross_market": """你是一位跨市场金融分析专家。请分析以下A股操纵案例与加密货币市场操纵的异同。

## A股操纵案例
- 股票：{stock_name}（{stock_code}）
- 操纵手法：{methods}
- 案情描述：{description}

## 请回答：

### 1. 跨市场模式对比
此类操纵手法在加密货币市场中是否存在类似的模式？具体表现有何异同？

### 2. 迁移学习价值
从加密货币数据中预训练的时序编码器，能捕捉到哪些对A股操纵检测有用的通用模式？

### 3. 跨市场特征共性
列出3个在两个市场中都适用的操纵检测特征。

### 4. 迁移限制
哪些A股特有的操纵特征无法从加密货币市场迁移？"""
}


QUALITY_EVAL_PROMPT = """请对以下市场操纵检测推理链（CoT）进行质量评估。按1-5分评分（5分最高）。

## 推理链内容：
{cot_text}

## 评分维度（每维度1-5分）：
1. **逻辑连贯性**：推理步骤是否连贯，前后是否矛盾？
2. **事实准确性**：金融知识和市场机制描述是否准确？
3. **可操作性**：分析结论是否可以转化为具体的检测规则/特征？
4. **完整性**：是否覆盖了案例的主要操纵特征？

请输出JSON格式：
{{"logical_coherence": X, "factual_accuracy": X, "actionability": X, "completeness": X, "average": X, "brief_comment": "..."}}"""


class CoTGenerator:
    """Batch CoT generator using Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        output_dir: str = "outputs/cot",
        rpm_limit: int = 10,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        self.model_name = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rpm_limit = rpm_limit
        self.delay = 60.0 / rpm_limit

        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate_for_case(
        self, case: dict, prompt_types: list[str] = None
    ) -> list[dict]:
        """Generate CoT entries for a single case."""
        prompt_types = prompt_types or ["primary", "counterfactual", "feature_analysis", "cross_market"]
        results = []

        fill_kwargs = self._case_to_fill_kwargs(case)

        for ptype in prompt_types:
            template = COT_PROMPTS.get(ptype)
            if not template:
                continue

            prompt = template.format(**fill_kwargs)
            logger.info(f"  Generating [{ptype}] for {case.get('doc_id', '?')}")

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                cot_text = response.text

                entry = {
                    "doc_id": case.get("doc_id"),
                    "stock_code": case.get("stock_code"),
                    "stock_name": case.get("stock_name"),
                    "prompt_type": ptype,
                    "cot_text": cot_text,
                    "model": self.model_name,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                results.append(entry)

            except Exception as e:
                logger.error(f"  Failed [{ptype}] for {case.get('doc_id')}: {e}")
                results.append({
                    "doc_id": case.get("doc_id"),
                    "prompt_type": ptype,
                    "error": str(e),
                })

            time.sleep(self.delay)

        return results

    def evaluate_quality(self, cot_entry: dict) -> dict:
        """Evaluate a single CoT entry using Gemini as judge."""
        if "error" in cot_entry or not cot_entry.get("cot_text"):
            return {"average": 0, "error": "no cot_text"}

        prompt = QUALITY_EVAL_PROMPT.format(cot_text=cot_entry["cot_text"][:3000])

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = response.text

            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]+\}', text)
            if json_match:
                scores = json.loads(json_match.group())
                return scores
            else:
                return {"average": 0, "error": "no JSON in response"}

        except Exception as e:
            logger.error(f"  Quality eval failed: {e}")
            return {"average": 0, "error": str(e)}

    def batch_generate(self, cases: list[dict]) -> list[dict]:
        """Generate CoTs for all cases."""
        all_results = []
        for i, case in enumerate(cases):
            logger.info(f"Processing case {i+1}/{len(cases)}: {case.get('doc_id')}")
            results = self.generate_for_case(case)
            all_results.extend(results)

            # Save incrementally
            self._save_results(all_results)

        return all_results

    def batch_evaluate(self, results: list[dict]) -> list[dict]:
        """Evaluate all CoT entries."""
        for i, entry in enumerate(results):
            if "error" in entry:
                entry["quality"] = {"average": 0}
                continue
            logger.info(f"Evaluating {i+1}/{len(results)}: {entry.get('doc_id')} [{entry.get('prompt_type')}]")
            scores = self.evaluate_quality(entry)
            entry["quality"] = scores
            time.sleep(self.delay)

        self._save_results(results)
        return results

    def _case_to_fill_kwargs(self, case: dict) -> dict:
        """Convert case dict to prompt template kwargs."""
        start = case.get("manipulation_start_date", "未知")
        end = case.get("manipulation_end_date", "未知")
        duration = case.get("manipulation_duration_days", "未知")

        methods = case.get("manipulation_methods", [])
        if not methods:
            methods = case.get("methods", [])
        methods_str = "、".join(methods) if methods else "未知"

        desc = case.get("manipulation_description", "")
        if not desc:
            paras = case.get("key_paragraphs", [])
            desc = " ".join(paras[:2]) if paras else "无详细描述"

        fine = case.get("fine_amount", 0)
        confiscation = case.get("confiscation_amount", 0)
        fine_info = f"罚款 {fine/1e4:.1f} 万元" if fine else ""
        if confiscation:
            fine_info += f"，没收违法所得 {confiscation/1e4:.1f} 万元"
        if not fine_info:
            fine_info = "详见处罚决定书"

        return {
            "stock_name": case.get("stock_name", "未知"),
            "stock_code": case.get("stock_code", "未知"),
            "start_date": start,
            "end_date": end,
            "duration": duration,
            "methods": methods_str,
            "description": desc[:800],
            "fine_info": fine_info,
        }

    def _save_results(self, results: list[dict]):
        out_path = self.output_dir / "cot_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def get_summary(self, results: list[dict]) -> dict:
        """Generate summary statistics."""
        total = len(results)
        errors = sum(1 for r in results if "error" in r)
        valid = [r for r in results if "error" not in r and "quality" in r]

        scores = [r["quality"].get("average", 0) for r in valid if r["quality"].get("average", 0) > 0]

        return {
            "total_generated": total,
            "errors": errors,
            "evaluated": len(valid),
            "avg_quality": sum(scores) / len(scores) if scores else 0,
            "min_quality": min(scores) if scores else 0,
            "max_quality": max(scores) if scores else 0,
            "above_3_5": sum(1 for s in scores if s >= 3.5),
        }
