"""
处罚决定书信息提取模块

使用大语言模型从处罚决定书中提取结构化信息。
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config


@dataclass
class ManipulationCase:
    """操纵案件结构化信息"""
    # 基础信息
    doc_id: str                           # 文书编号
    case_number: str                      # 案件编号
    defendant_name: str                   # 被处罚人/机构名称
    defendant_type: str                   # 被处罚人类型（个人/机构）
    
    # 股票信息
    stock_code: str                       # 股票代码
    stock_name: str                       # 股票名称
    
    # 操纵时间
    manipulation_start_date: str          # 操纵开始日期
    manipulation_end_date: str            # 操纵结束日期
    manipulation_duration_days: int       # 操纵持续天数
    
    # 操纵手法
    manipulation_methods: List[str]       # 操纵手法列表
    manipulation_description: str         # 操纵行为描述
    
    # 账户信息
    account_count: int                    # 涉案账户数量
    account_group_description: str        # 账户组描述
    
    # 资金和收益
    involved_amount: float                # 涉案金额（元）
    illegal_profit: float                 # 违法所得（元）
    
    # 处罚信息
    penalty_type: str                     # 处罚类型
    fine_amount: float                    # 罚款金额（元）
    confiscation_amount: float            # 没收金额（元）
    market_ban_years: int                 # 市场禁入年限
    
    # 元数据
    source_url: str                       # 原文链接
    publish_date: str                     # 发布日期
    extract_time: str                     # 提取时间
    confidence_score: float               # 提取置信度
    
    # 原文片段
    key_paragraphs: List[str] = field(default_factory=list)  # 关键段落


# 信息提取提示词模板
EXTRACTION_PROMPT_TEMPLATE = """你是一位专业的金融法律文书分析专家。请从以下证监会行政处罚决定书中提取结构化信息。

## 待分析文本：
{content}

## 提取要求：
请严格按照以下JSON格式输出提取结果。如果某个字段无法从文本中提取，请填写null。

```json
{{
    "case_number": "案件编号，如〔2023〕1号",
    "defendant_name": "被处罚人或机构的全称",
    "defendant_type": "个人或机构",
    
    "stock_code": "6位股票代码，如600000",
    "stock_name": "股票简称",
    
    "manipulation_start_date": "操纵开始日期，格式YYYY-MM-DD",
    "manipulation_end_date": "操纵结束日期，格式YYYY-MM-DD",
    
    "manipulation_methods": ["操纵手法1", "操纵手法2"],
    "manipulation_description": "简要描述操纵行为（100字以内）",
    
    "account_count": "涉案账户数量，整数",
    "account_group_description": "账户组特征描述（50字以内）",
    
    "involved_amount": "涉案金额，单位元，数字",
    "illegal_profit": "违法所得，单位元，数字",
    
    "penalty_type": "警告/罚款/没收/市场禁入等",
    "fine_amount": "罚款金额，单位元，数字",
    "confiscation_amount": "没收金额，单位元，数字",
    "market_ban_years": "市场禁入年限，整数，0表示无禁入",
    
    "key_paragraphs": ["包含关键事实的原文段落1", "段落2"]
}}
```

## 注意事项：
1. 金额请统一换算为元（人民币），不要带单位
2. 日期格式统一为YYYY-MM-DD
3. 操纵手法请使用标准术语：连续交易、对倒交易、虚假申报、约定交易、洗售、尾市拉抬等
4. 如果存在多只股票，只提取主要操纵对象
5. key_paragraphs只保留最关键的2-3段原文

请直接输出JSON，不要有其他内容。"""


class PenaltyExtractor:
    """处罚决定书信息提取器"""
    
    def __init__(self, llm_provider: str = None):
        """
        初始化提取器
        
        Args:
            llm_provider: LLM服务提供商（默认从配置读取）
        """
        self.llm_config = config.config.get("llm", {})
        self.provider = llm_provider or self.llm_config.get("provider", "gemini")
        self.model = self.llm_config.get("model", "gemini-1.5-flash")
        self.temperature = self.llm_config.get("temperature", 0.1)
        self.max_tokens = self.llm_config.get("max_tokens", 4096)
        
        # 初始化LLM客户端
        self.client = self._init_client()
        
        # 输出目录
        self.output_dir = Path(config.paths_config.get("processed_data", "data/processed")) / "extracted_cases"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"信息提取器初始化完成，使用模型: {self.provider}/{self.model}")
    
    def _init_client(self):
        """初始化LLM客户端"""
        # 根据provider获取对应的API密钥
        if self.provider == "gemini":
            api_key = config.get_env("GOOGLE_API_KEY")
        else:
            api_key = config.get_api_key(self.provider)
        
        if not api_key:
            log.warning(f"未找到{self.provider}的API密钥，将使用模拟模式")
            log.warning("请设置环境变量: export GOOGLE_API_KEY='your_key'")
            return None
        
        if self.provider == "gemini":
            try:
                # 使用新版google-genai SDK
                from google import genai
                client = genai.Client(api_key=api_key)
                return client
            except ImportError:
                # 回退到旧版SDK
                try:
                    import google.generativeai as genai_old
                    genai_old.configure(api_key=api_key)
                    return ("old", genai_old.GenerativeModel(self.model))
                except ImportError:
                    log.error("请安装: pip install google-genai")
                    return None
        elif self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=api_key)
        else:
            log.warning(f"不支持的LLM提供商: {self.provider}")
            return None
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        调用LLM
        
        Args:
            prompt: 提示词
            
        Returns:
            LLM响应文本
        """
        if not self.client:
            log.warning("LLM客户端未初始化，使用模拟响应")
            return self._mock_response()
        
        try:
            if self.provider == "gemini":
                # 检查是新版还是旧版SDK
                if isinstance(self.client, tuple) and self.client[0] == "old":
                    # 旧版SDK
                    response = self.client[1].generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens,
                        }
                    )
                    return response.text
                else:
                    # 新版SDK (google-genai)
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens,
                        }
                    )
                    return response.text
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一位专业的金融法律文书分析专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
                
        except Exception as e:
            log.error(f"LLM调用失败: {e}")
            return None
    
    def _mock_response(self) -> str:
        """生成模拟响应（用于测试）"""
        return json.dumps({
            "case_number": "〔2023〕1号",
            "defendant_name": "测试被处罚人",
            "defendant_type": "个人",
            "stock_code": "000001",
            "stock_name": "测试股票",
            "manipulation_start_date": "2022-01-01",
            "manipulation_end_date": "2022-06-30",
            "manipulation_methods": ["连续交易", "对倒交易"],
            "manipulation_description": "通过关联账户进行对倒交易操纵股价",
            "account_count": 10,
            "account_group_description": "家族关联账户组",
            "involved_amount": 100000000,
            "illegal_profit": 5000000,
            "penalty_type": "罚款",
            "fine_amount": 10000000,
            "confiscation_amount": 5000000,
            "market_ban_years": 5,
            "key_paragraphs": ["测试关键段落"]
        }, ensure_ascii=False)
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """
        解析LLM响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析后的字典
        """
        if not response:
            return None
        
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return json.loads(response)
        except json.JSONDecodeError as e:
            log.error(f"JSON解析失败: {e}")
            return None
    
    def _clean_content(self, content: str, max_length: int = 8000) -> str:
        """
        清理和截断内容
        
        Args:
            content: 原始内容
            max_length: 最大长度
            
        Returns:
            清理后的内容
        """
        # 移除多余空白
        content = re.sub(r'\s+', ' ', content)
        
        # 移除特殊字符
        content = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。、；：""''（）【】《》\-\.\s]', '', content)
        
        # 截断
        if len(content) > max_length:
            # 保留开头和结尾
            head_len = max_length * 2 // 3
            tail_len = max_length // 3
            content = content[:head_len] + "\n...[中间部分省略]...\n" + content[-tail_len:]
        
        return content
    
    def _calculate_confidence(self, extracted: Dict) -> float:
        """
        计算提取置信度
        
        Args:
            extracted: 提取结果
            
        Returns:
            置信度（0-1）
        """
        # 关键字段列表
        key_fields = [
            "stock_code", "stock_name",
            "manipulation_start_date", "manipulation_end_date",
            "manipulation_methods", "manipulation_description",
            "defendant_name"
        ]
        
        # 计算非空字段比例
        valid_count = sum(1 for f in key_fields if extracted.get(f))
        confidence = valid_count / len(key_fields)
        
        # 额外检查
        if extracted.get("stock_code") and not re.match(r'^\d{6}$', str(extracted.get("stock_code", ""))):
            confidence -= 0.1
        
        return max(0, min(1, confidence))
    
    def extract_from_document(self, doc_path: str) -> Optional[ManipulationCase]:
        """
        从单个文档提取信息
        
        Args:
            doc_path: 文档路径
            
        Returns:
            提取的案件信息
        """
        doc_path = Path(doc_path)
        
        if not doc_path.exists():
            log.error(f"文档不存在: {doc_path}")
            return None
        
        # 读取文档
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        content = doc_data.get("content", "")
        if not content:
            log.warning(f"文档内容为空: {doc_path}")
            return None
        
        log.info(f"提取文档: {doc_data.get('title', doc_path.name)}")
        
        # 清理内容
        cleaned_content = self._clean_content(content)
        
        # 构建提示词
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(content=cleaned_content)
        
        # 调用LLM
        response = self._call_llm(prompt)
        if not response:
            return None
        
        # 解析响应
        extracted = self._parse_llm_response(response)
        if not extracted:
            return None
        
        # 计算置信度
        confidence = self._calculate_confidence(extracted)
        
        # 构建案件对象
        try:
            case = ManipulationCase(
                doc_id=doc_data.get("doc_id", doc_path.stem),
                case_number=extracted.get("case_number", ""),
                defendant_name=extracted.get("defendant_name", ""),
                defendant_type=extracted.get("defendant_type", ""),
                stock_code=str(extracted.get("stock_code", "")),
                stock_name=extracted.get("stock_name", ""),
                manipulation_start_date=extracted.get("manipulation_start_date", ""),
                manipulation_end_date=extracted.get("manipulation_end_date", ""),
                manipulation_duration_days=self._calc_duration(
                    extracted.get("manipulation_start_date"),
                    extracted.get("manipulation_end_date")
                ),
                manipulation_methods=extracted.get("manipulation_methods", []),
                manipulation_description=extracted.get("manipulation_description", ""),
                account_count=int(extracted.get("account_count") or 0),
                account_group_description=extracted.get("account_group_description", ""),
                involved_amount=float(extracted.get("involved_amount") or 0),
                illegal_profit=float(extracted.get("illegal_profit") or 0),
                penalty_type=extracted.get("penalty_type", ""),
                fine_amount=float(extracted.get("fine_amount") or 0),
                confiscation_amount=float(extracted.get("confiscation_amount") or 0),
                market_ban_years=int(extracted.get("market_ban_years") or 0),
                source_url=doc_data.get("url", ""),
                publish_date=doc_data.get("publish_date", ""),
                extract_time=datetime.now().isoformat(),
                confidence_score=confidence,
                key_paragraphs=extracted.get("key_paragraphs", [])
            )
            
            return case
            
        except Exception as e:
            log.error(f"构建案件对象失败: {e}")
            return None
    
    def _calc_duration(self, start_date: str, end_date: str) -> int:
        """计算操纵持续天数"""
        try:
            if not start_date or not end_date:
                return 0
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            return (end - start).days + 1
        except:
            return 0
    
    def save_case(self, case: ManipulationCase) -> str:
        """
        保存提取的案件
        
        Args:
            case: 案件对象
            
        Returns:
            保存路径
        """
        file_path = self.output_dir / f"{case.doc_id}_extracted.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(case), f, ensure_ascii=False, indent=2)
        
        log.info(f"案件已保存: {file_path}")
        return str(file_path)
    
    def batch_extract(self, input_dir: str, min_confidence: float = 0.5) -> List[ManipulationCase]:
        """
        批量提取
        
        Args:
            input_dir: 输入目录
            min_confidence: 最小置信度阈值
            
        Returns:
            提取的案件列表
        """
        input_path = Path(input_dir)
        doc_files = list(input_path.rglob("*.json"))
        
        log.info(f"开始批量提取，共 {len(doc_files)} 个文档")
        
        cases = []
        for doc_file in tqdm(doc_files, desc="提取进度"):
            # 跳过索引文件
            if doc_file.name == "index.json":
                continue
            
            case = self.extract_from_document(str(doc_file))
            
            if case and case.confidence_score >= min_confidence:
                self.save_case(case)
                cases.append(case)
            elif case:
                log.warning(f"置信度过低，跳过: {doc_file.name}, 置信度: {case.confidence_score:.2f}")
        
        # 保存汇总
        self._save_summary(cases)
        
        log.info(f"批量提取完成，有效案件: {len(cases)}")
        return cases
    
    def _save_summary(self, cases: List[ManipulationCase]):
        """保存汇总信息"""
        summary = {
            "total_cases": len(cases),
            "extract_time": datetime.now().isoformat(),
            "cases": [
                {
                    "doc_id": c.doc_id,
                    "stock_code": c.stock_code,
                    "stock_name": c.stock_name,
                    "manipulation_start_date": c.manipulation_start_date,
                    "manipulation_end_date": c.manipulation_end_date,
                    "methods": c.manipulation_methods,
                    "confidence": c.confidence_score
                }
                for c in cases
            ]
        }
        
        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    extractor = PenaltyExtractor()
    
    # 批量提取
    input_dir = "data/raw/csrc_penalties"
    cases = extractor.batch_extract(input_dir)
    
    print(f"\n提取完成！共提取 {len(cases)} 个有效案件")
    print(f"结果保存在: {extractor.output_dir}")


if __name__ == "__main__":
    main()

