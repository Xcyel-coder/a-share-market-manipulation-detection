"""
数据质量检查工具

验证采集和标注的数据质量，生成质量报告。
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config


@dataclass
class QualityReport:
    """质量报告"""
    report_id: str
    report_time: str
    data_type: str              # csrc / astock / crypto
    total_records: int
    valid_records: int
    invalid_records: int
    validity_rate: float
    
    # 完整性指标
    missing_fields: Dict[str, int]
    completeness_rate: float
    
    # 一致性指标
    consistency_issues: List[Dict]
    consistency_rate: float
    
    # 异常值指标
    outlier_count: int
    outlier_rate: float
    
    # 详细问题
    issues: List[Dict]
    
    # 建议
    recommendations: List[str]


class DataValidator:
    """数据质量验证器"""
    
    def __init__(self):
        """初始化"""
        self.data_dir = Path(config.paths_config.get("raw_data", "data/raw"))
        self.processed_dir = Path(config.paths_config.get("processed_data", "data/processed"))
        self.labeled_dir = Path(config.paths_config.get("labeled_data", "data/labeled"))
        
        # 报告输出目录
        self.report_dir = Path("data/quality_reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("数据质量验证器初始化完成")
    
    def validate_csrc_data(self) -> QualityReport:
        """验证证监会处罚数据"""
        log.info("验证证监会处罚数据...")
        
        csrc_dir = self.data_dir / "csrc_penalties"
        if not csrc_dir.exists():
            return self._empty_report("csrc", "数据目录不存在")
        
        # 收集所有文档
        doc_files = list(csrc_dir.rglob("*.json"))
        doc_files = [f for f in doc_files if f.name != "index.json"]
        
        if not doc_files:
            return self._empty_report("csrc", "无数据文件")
        
        total = len(doc_files)
        valid = 0
        issues = []
        missing_fields = {
            "doc_id": 0, "title": 0, "content": 0,
            "publish_date": 0, "url": 0
        }
        
        for doc_file in tqdm(doc_files, desc="验证CSRC数据"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                is_valid = True
                
                # 检查必需字段
                for field in missing_fields.keys():
                    if not doc.get(field):
                        missing_fields[field] += 1
                        is_valid = False
                
                # 检查内容长度
                content = doc.get("content", "")
                if len(content) < 500:
                    issues.append({
                        "file": str(doc_file.name),
                        "issue": "content_too_short",
                        "detail": f"内容长度仅{len(content)}字符"
                    })
                    is_valid = False
                
                # 检查关键词
                keywords = doc.get("keywords_matched", [])
                if not keywords:
                    issues.append({
                        "file": str(doc_file.name),
                        "issue": "no_keywords",
                        "detail": "未匹配到操纵关键词"
                    })
                
                if is_valid:
                    valid += 1
                    
            except Exception as e:
                issues.append({
                    "file": str(doc_file.name),
                    "issue": "parse_error",
                    "detail": str(e)
                })
        
        # 计算指标
        completeness = 1 - sum(missing_fields.values()) / (total * len(missing_fields))
        
        report = QualityReport(
            report_id=f"csrc_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_time=datetime.now().isoformat(),
            data_type="csrc",
            total_records=total,
            valid_records=valid,
            invalid_records=total - valid,
            validity_rate=valid / total if total > 0 else 0,
            missing_fields=missing_fields,
            completeness_rate=completeness,
            consistency_issues=[],
            consistency_rate=1.0,
            outlier_count=0,
            outlier_rate=0,
            issues=issues[:100],  # 只保留前100个问题
            recommendations=self._generate_csrc_recommendations(missing_fields, issues)
        )
        
        return report
    
    def validate_extracted_cases(self) -> QualityReport:
        """验证提取的案件数据"""
        log.info("验证提取的案件数据...")
        
        cases_dir = self.processed_dir / "extracted_cases"
        if not cases_dir.exists():
            return self._empty_report("extracted_cases", "数据目录不存在")
        
        case_files = list(cases_dir.glob("*_extracted.json"))
        
        if not case_files:
            return self._empty_report("extracted_cases", "无数据文件")
        
        total = len(case_files)
        valid = 0
        issues = []
        consistency_issues = []
        
        required_fields = [
            "stock_code", "stock_name", "manipulation_start_date",
            "manipulation_end_date", "manipulation_methods"
        ]
        missing_fields = {f: 0 for f in required_fields}
        
        for case_file in tqdm(case_files, desc="验证提取数据"):
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case = json.load(f)
                
                is_valid = True
                
                # 检查必需字段
                for field in required_fields:
                    value = case.get(field)
                    if not value or value == "null":
                        missing_fields[field] += 1
                        is_valid = False
                
                # 检查股票代码格式
                stock_code = case.get("stock_code", "")
                if stock_code and not self._is_valid_stock_code(stock_code):
                    issues.append({
                        "file": case_file.name,
                        "issue": "invalid_stock_code",
                        "detail": f"无效股票代码: {stock_code}"
                    })
                    is_valid = False
                
                # 检查日期逻辑
                start_date = case.get("manipulation_start_date")
                end_date = case.get("manipulation_end_date")
                if start_date and end_date:
                    if start_date > end_date:
                        consistency_issues.append({
                            "file": case_file.name,
                            "issue": "date_inconsistency",
                            "detail": f"开始日期({start_date})晚于结束日期({end_date})"
                        })
                        is_valid = False
                
                # 检查置信度
                confidence = case.get("confidence_score", 0)
                if confidence < 0.5:
                    issues.append({
                        "file": case_file.name,
                        "issue": "low_confidence",
                        "detail": f"置信度过低: {confidence:.2f}"
                    })
                
                if is_valid:
                    valid += 1
                    
            except Exception as e:
                issues.append({
                    "file": case_file.name,
                    "issue": "parse_error",
                    "detail": str(e)
                })
        
        completeness = 1 - sum(missing_fields.values()) / (total * len(required_fields))
        consistency = 1 - len(consistency_issues) / total if total > 0 else 1
        
        report = QualityReport(
            report_id=f"extracted_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_time=datetime.now().isoformat(),
            data_type="extracted_cases",
            total_records=total,
            valid_records=valid,
            invalid_records=total - valid,
            validity_rate=valid / total if total > 0 else 0,
            missing_fields=missing_fields,
            completeness_rate=completeness,
            consistency_issues=consistency_issues[:50],
            consistency_rate=consistency,
            outlier_count=0,
            outlier_rate=0,
            issues=issues[:100],
            recommendations=self._generate_extraction_recommendations(missing_fields, issues)
        )
        
        return report
    
    def validate_crypto_data(self) -> QualityReport:
        """验证加密货币数据"""
        log.info("验证加密货币数据...")
        
        crypto_dir = self.data_dir / "crypto"
        if not crypto_dir.exists():
            return self._empty_report("crypto", "数据目录不存在")
        
        symbol_dirs = [d for d in crypto_dir.iterdir() if d.is_dir()]
        
        if not symbol_dirs:
            return self._empty_report("crypto", "无数据目录")
        
        total_records = 0
        valid_records = 0
        issues = []
        outlier_count = 0
        
        for symbol_dir in tqdm(symbol_dirs, desc="验证加密货币数据"):
            parquet_files = list(symbol_dir.glob("*.parquet"))
            
            for pf in parquet_files:
                try:
                    df = pd.read_parquet(pf)
                    total_records += len(df)
                    
                    # 检查缺失值
                    missing = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum()
                    if missing > 0:
                        issues.append({
                            "file": str(pf),
                            "issue": "missing_values",
                            "detail": f"缺失值数量: {missing}"
                        })
                    
                    # 检查价格逻辑（high >= low）
                    invalid_ohlc = (df['high'] < df['low']).sum()
                    if invalid_ohlc > 0:
                        issues.append({
                            "file": str(pf),
                            "issue": "invalid_ohlc",
                            "detail": f"无效OHLC记录: {invalid_ohlc}"
                        })
                    
                    # 检查异常值（使用Z分数）
                    for col in ['close', 'volume']:
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        outliers = (z_scores > 5).sum()
                        outlier_count += outliers
                    
                    valid_records += len(df) - missing - invalid_ohlc
                    
                except Exception as e:
                    issues.append({
                        "file": str(pf),
                        "issue": "read_error",
                        "detail": str(e)
                    })
        
        report = QualityReport(
            report_id=f"crypto_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_time=datetime.now().isoformat(),
            data_type="crypto",
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=total_records - valid_records,
            validity_rate=valid_records / total_records if total_records > 0 else 0,
            missing_fields={},
            completeness_rate=valid_records / total_records if total_records > 0 else 0,
            consistency_issues=[],
            consistency_rate=1.0,
            outlier_count=outlier_count,
            outlier_rate=outlier_count / (total_records * 2) if total_records > 0 else 0,
            issues=issues[:100],
            recommendations=self._generate_crypto_recommendations(issues, outlier_count)
        )
        
        return report
    
    def validate_crypto_labels(self) -> QualityReport:
        """验证加密货币标注数据"""
        log.info("验证加密货币标注数据...")
        
        labeled_dir = self.labeled_dir / "crypto"
        if not labeled_dir.exists():
            return self._empty_report("crypto_labels", "数据目录不存在")
        
        symbol_dirs = [d for d in labeled_dir.iterdir() if d.is_dir()]
        
        if not symbol_dirs:
            return self._empty_report("crypto_labels", "无数据目录")
        
        total_labels = 0
        valid_labels = 0
        issues = []
        
        confidence_values = []
        
        for symbol_dir in tqdm(symbol_dirs, desc="验证标注数据"):
            labels_file = symbol_dir / "labels.json"
            
            if not labels_file.exists():
                continue
            
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                
                for label in labels:
                    total_labels += 1
                    
                    # 检查必需字段
                    if not all([
                        label.get("pump_return"),
                        label.get("dump_return"),
                        label.get("confidence")
                    ]):
                        issues.append({
                            "file": str(labels_file),
                            "label_id": label.get("label_id"),
                            "issue": "missing_fields"
                        })
                        continue
                    
                    # 收集置信度
                    confidence_values.append(label.get("confidence", 0))
                    
                    # 检查涨幅合理性
                    pump_return = label.get("pump_return", 0)
                    if pump_return > 10:  # 涨幅超过1000%可能是异常
                        issues.append({
                            "file": str(labels_file),
                            "label_id": label.get("label_id"),
                            "issue": "extreme_pump_return",
                            "detail": f"涨幅: {pump_return:.0%}"
                        })
                    
                    valid_labels += 1
                    
            except Exception as e:
                issues.append({
                    "file": str(labels_file),
                    "issue": "parse_error",
                    "detail": str(e)
                })
        
        # 计算置信度分布
        avg_confidence = np.mean(confidence_values) if confidence_values else 0
        
        report = QualityReport(
            report_id=f"crypto_labels_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_time=datetime.now().isoformat(),
            data_type="crypto_labels",
            total_records=total_labels,
            valid_records=valid_labels,
            invalid_records=total_labels - valid_labels,
            validity_rate=valid_labels / total_labels if total_labels > 0 else 0,
            missing_fields={},
            completeness_rate=valid_labels / total_labels if total_labels > 0 else 0,
            consistency_issues=[],
            consistency_rate=1.0,
            outlier_count=len([c for c in confidence_values if c < 0.5]),
            outlier_rate=len([c for c in confidence_values if c < 0.5]) / len(confidence_values) if confidence_values else 0,
            issues=issues[:100],
            recommendations=[
                f"平均置信度: {avg_confidence:.2f}",
                "建议剔除置信度低于0.6的样本",
                "建议人工抽查极端涨幅样本"
            ]
        )
        
        return report
    
    def _is_valid_stock_code(self, code: str) -> bool:
        """检查股票代码是否有效"""
        code = str(code).strip()
        if len(code) != 6:
            return False
        if not code.isdigit():
            return False
        # 检查前缀
        valid_prefixes = ['00', '30', '60', '68', '83', '87']
        return any(code.startswith(p) for p in valid_prefixes)
    
    def _empty_report(self, data_type: str, reason: str) -> QualityReport:
        """生成空报告"""
        return QualityReport(
            report_id=f"{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_time=datetime.now().isoformat(),
            data_type=data_type,
            total_records=0,
            valid_records=0,
            invalid_records=0,
            validity_rate=0,
            missing_fields={},
            completeness_rate=0,
            consistency_issues=[],
            consistency_rate=0,
            outlier_count=0,
            outlier_rate=0,
            issues=[{"issue": "no_data", "detail": reason}],
            recommendations=[f"请先执行数据采集: {reason}"]
        )
    
    def _generate_csrc_recommendations(
        self,
        missing_fields: Dict,
        issues: List
    ) -> List[str]:
        """生成CSRC数据建议"""
        recs = []
        
        if missing_fields.get("content", 0) > 0:
            recs.append(f"有{missing_fields['content']}个文档内容为空，建议重新爬取")
        
        if missing_fields.get("publish_date", 0) > 0:
            recs.append("部分文档缺少发布日期，建议从正文提取或补充")
        
        short_content = len([i for i in issues if i.get("issue") == "content_too_short"])
        if short_content > 0:
            recs.append(f"有{short_content}个文档内容过短，可能是爬取不完整")
        
        return recs
    
    def _generate_extraction_recommendations(
        self,
        missing_fields: Dict,
        issues: List
    ) -> List[str]:
        """生成提取数据建议"""
        recs = []
        
        if missing_fields.get("stock_code", 0) > 0:
            recs.append(f"有{missing_fields['stock_code']}个案件未提取到股票代码，建议优化提示词")
        
        if missing_fields.get("manipulation_methods", 0) > 0:
            recs.append("部分案件未识别操纵手法，建议检查提示词设计")
        
        low_conf = len([i for i in issues if i.get("issue") == "low_confidence"])
        if low_conf > 0:
            recs.append(f"有{low_conf}个案件置信度较低，建议人工复核")
        
        return recs
    
    def _generate_crypto_recommendations(
        self,
        issues: List,
        outlier_count: int
    ) -> List[str]:
        """生成加密货币数据建议"""
        recs = []
        
        missing_issues = len([i for i in issues if i.get("issue") == "missing_values"])
        if missing_issues > 0:
            recs.append(f"有{missing_issues}个文件存在缺失值，建议使用插值填充")
        
        if outlier_count > 0:
            recs.append(f"发现{outlier_count}个异常值点，建议进行异常值处理")
        
        return recs
    
    def save_report(self, report: QualityReport) -> str:
        """保存报告"""
        file_path = self.report_dir / f"{report.report_id}.json"
        
        def _convert(obj):
            """处理 numpy 类型的 JSON 序列化"""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=_convert)
        
        log.info(f"报告已保存: {file_path}")
        return str(file_path)
    
    def run_full_validation(self) -> Dict[str, QualityReport]:
        """运行完整验证"""
        log.info("开始完整数据质量验证...")
        
        reports = {}
        
        # 验证各类数据
        reports["csrc"] = self.validate_csrc_data()
        reports["extracted_cases"] = self.validate_extracted_cases()
        reports["crypto"] = self.validate_crypto_data()
        reports["crypto_labels"] = self.validate_crypto_labels()
        
        # 保存所有报告
        for name, report in reports.items():
            self.save_report(report)
        
        # 生成汇总
        summary = {
            "validation_time": datetime.now().isoformat(),
            "reports": {
                name: {
                    "total_records": r.total_records,
                    "validity_rate": f"{r.validity_rate:.1%}",
                    "completeness_rate": f"{r.completeness_rate:.1%}",
                    "issues_count": len(r.issues)
                }
                for name, r in reports.items()
            }
        }
        
        summary_path = self.report_dir / "validation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        log.info("完整验证完成")
        return reports
    
    def print_summary(self, reports: Dict[str, QualityReport]):
        """打印验证摘要"""
        print("\n" + "=" * 60)
        print("数据质量验证摘要")
        print("=" * 60)
        
        for name, report in reports.items():
            print(f"\n【{report.data_type}】")
            print(f"  总记录数: {report.total_records}")
            print(f"  有效率: {report.validity_rate:.1%}")
            print(f"  完整性: {report.completeness_rate:.1%}")
            print(f"  问题数: {len(report.issues)}")
            
            if report.recommendations:
                print("  建议:")
                for rec in report.recommendations[:3]:
                    print(f"    - {rec}")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    validator = DataValidator()
    
    # 运行完整验证
    reports = validator.run_full_validation()
    
    # 打印摘要
    validator.print_summary(reports)
    
    print(f"\n详细报告保存在: {validator.report_dir}")


if __name__ == "__main__":
    main()

