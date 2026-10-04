"""
证监会处罚决定书手动导入工具

当自动爬虫因反爬/JS渲染等原因无法工作时，提供手动导入文书的能力。
支持三种导入方式：
1. 从文本文件批量导入
2. 从剪贴板单条导入
3. 从已下载的HTML/PDF文件导入

使用场景：在浏览器中手动访问证监会官网 → 复制处罚决定书全文 → 使用本工具导入
"""
import json
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config


MANIPULATION_KEYWORDS = [
    "操纵", "对倒", "对敲", "拉抬", "打压",
    "虚假申报", "撤单", "洗售", "相对委托",
    "尾市", "开盘", "连续交易", "约定交易"
]


@dataclass
class PenaltyDocument:
    """处罚决定书数据结构"""
    doc_id: str
    title: str
    url: str
    publish_date: str
    content: str
    case_type: str
    keywords_matched: List[str]
    crawl_time: str


def extract_doc_id(title: str, content: str) -> str:
    """从标题或内容中提取文书编号"""
    patterns = [
        r"〔(\d{4})〕(\d+)号",
        r"\[(\d{4})\](\d+)号",
        r"（(\d{4})）(\d+)号",
    ]
    for text in [title, content[:500]]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                year, num = match.groups()
                return f"CSRC-{year}-{num.zfill(3)}"
    return f"CSRC-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def extract_publish_date(content: str) -> str:
    """从内容中提取发布日期（通常在文末）"""
    # 优先匹配文末的日期（"中国证监会\n2024年2月28日"）
    patterns = [
        r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*$",
        r"(\d{4})年(\d{1,2})月(\d{1,2})日",
        r"(\d{4})-(\d{2})-(\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            g = match.groups()
            return f"{g[0]}-{g[1].zfill(2)}-{g[2].zfill(2)}"
    return datetime.now().strftime("%Y-%m-%d")


def detect_keywords(text: str) -> List[str]:
    """检测操纵类关键词"""
    return [kw for kw in MANIPULATION_KEYWORDS if kw in text]


def classify_case(keywords: List[str], content: str) -> str:
    """判断案件类型"""
    manipulation_kws = {"操纵", "对倒", "拉抬", "打压", "连续交易", "虚假申报", "洗售"}
    if any(kw in manipulation_kws for kw in keywords):
        return "操纵证券市场"
    if "内幕" in content:
        return "内幕交易"
    if "信息披露" in content or "虚假陈述" in content:
        return "信息披露违规"
    return "其他证券违法"


class CSRCManualImporter:
    """证监会处罚决定书手动导入工具"""
    
    def __init__(self):
        raw_path = config.paths_config.get("raw_data", "data/raw") if hasattr(config, 'paths_config') else "data/raw"
        self.output_dir = Path(raw_path) / "csrc_penalties"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"手动导入工具初始化，输出目录: {self.output_dir}")
    
    def import_from_text(
        self,
        content: str,
        title: str = "",
        url: str = "",
        only_manipulation: bool = True
    ) -> Optional[PenaltyDocument]:
        """
        从纯文本导入单条处罚决定书
        
        Args:
            content: 处罚决定书全文
            title: 标题（可选，自动从内容提取）
            url: 原文链接（可选）
            only_manipulation: 是否仅导入操纵类案件
            
        Returns:
            导入成功返回文档对象，否则None
        """
        content = content.strip()
        if len(content) < 200:
            log.warning("内容过短（<200字），跳过")
            return None
        
        # 自动提取标题
        if not title:
            first_line = content.split("\n")[0].strip()
            if "行政处罚" in first_line or "决定书" in first_line:
                title = first_line
            else:
                title = f"处罚决定书-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        keywords = detect_keywords(content)
        
        if only_manipulation and not keywords:
            log.info(f"非操纵类案件，跳过: {title[:30]}...")
            return None
        
        doc = PenaltyDocument(
            doc_id=extract_doc_id(title, content),
            title=title,
            url=url,
            publish_date=extract_publish_date(content),
            content=content,
            case_type=classify_case(keywords, content),
            keywords_matched=keywords,
            crawl_time=datetime.now().isoformat()
        )
        
        self._save_document(doc)
        return doc
    
    def import_from_file(self, file_path: str, **kwargs) -> Optional[PenaltyDocument]:
        """从文本文件导入"""
        path = Path(file_path)
        if not path.exists():
            log.error(f"文件不存在: {file_path}")
            return None
        
        content = path.read_text(encoding="utf-8")
        title = kwargs.get("title", path.stem)
        return self.import_from_text(content, title=title, **kwargs)
    
    def import_from_directory(
        self,
        dir_path: str,
        pattern: str = "*.txt",
        only_manipulation: bool = True
    ) -> List[PenaltyDocument]:
        """
        从目录批量导入
        
        将处罚决定书全文保存为 .txt 文件放入目录，然后调用此方法批量导入。
        
        Args:
            dir_path: 目录路径
            pattern: 文件匹配模式
            only_manipulation: 是否仅导入操纵类
            
        Returns:
            成功导入的文档列表
        """
        dir_p = Path(dir_path)
        if not dir_p.exists():
            log.error(f"目录不存在: {dir_path}")
            return []
        
        files = list(dir_p.glob(pattern))
        log.info(f"在 {dir_path} 中发现 {len(files)} 个文件")
        
        docs = []
        for f in sorted(files):
            doc = self.import_from_file(str(f), only_manipulation=only_manipulation)
            if doc:
                docs.append(doc)
                log.info(f"导入成功: {doc.doc_id} - {doc.title[:30]}")
        
        # 更新索引
        self._update_index()
        
        log.info(f"批量导入完成: {len(docs)}/{len(files)} 个文件")
        return docs
    
    def _save_document(self, doc: PenaltyDocument):
        """保存单个文档"""
        year = doc.publish_date[:4] if doc.publish_date else datetime.now().strftime("%Y")
        year_dir = self.output_dir / year
        year_dir.mkdir(exist_ok=True)
        
        file_path = year_dir / f"{doc.doc_id}.json"
        
        # 避免覆盖已有文件
        if file_path.exists():
            log.info(f"文件已存在，跳过: {file_path}")
            return
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(doc), f, ensure_ascii=False, indent=2)
        log.info(f"已保存: {file_path}")
    
    def _update_index(self):
        """更新索引文件"""
        all_docs = []
        for json_file in self.output_dir.rglob("*.json"):
            if json_file.name == "index.json":
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_docs.append({
                    "doc_id": data.get("doc_id"),
                    "title": data.get("title"),
                    "case_type": data.get("case_type"),
                    "publish_date": data.get("publish_date"),
                    "keywords_matched": data.get("keywords_matched", [])
                })
            except Exception:
                continue
        
        index = {
            "total": len(all_docs),
            "update_time": datetime.now().isoformat(),
            "manipulation_cases": len([d for d in all_docs if d.get("case_type") == "操纵证券市场"]),
            "files": all_docs
        }
        
        index_path = self.output_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        log.info(f"索引已更新: {len(all_docs)} 个文档")
    
    def get_stats(self) -> Dict:
        """获取当前数据统计"""
        index_path = self.output_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        self._update_index()
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    """交互式导入"""
    importer = CSRCManualImporter()
    
    print("=" * 60)
    print("证监会处罚决定书手动导入工具")
    print("=" * 60)
    print("\n选择导入方式:")
    print("  1. 从目录批量导入 (.txt 文件)")
    print("  2. 从单个文件导入")
    print("  3. 从剪贴板粘贴导入")
    print("  4. 查看当前数据统计")
    print("  0. 退出\n")
    
    while True:
        choice = input("请选择 [0-4]: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            dir_path = input("输入目录路径: ").strip()
            docs = importer.import_from_directory(dir_path)
            print(f"\n成功导入 {len(docs)} 个文档")
        elif choice == "2":
            file_path = input("输入文件路径: ").strip()
            doc = importer.import_from_file(file_path)
            if doc:
                print(f"\n导入成功: {doc.doc_id}")
            else:
                print("\n导入失败")
        elif choice == "3":
            print("请粘贴处罚决定书全文（输入 END 结束）:")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            content = "\n".join(lines)
            doc = importer.import_from_text(content)
            if doc:
                print(f"\n导入成功: {doc.doc_id}")
            else:
                print("\n导入失败（非操纵类或内容过短）")
        elif choice == "4":
            stats = importer.get_stats()
            print(f"\n当前数据统计:")
            print(f"  总文档数: {stats.get('total', 0)}")
            print(f"  操纵类: {stats.get('manipulation_cases', 0)}")
            print(f"  更新时间: {stats.get('update_time', 'N/A')}")
        else:
            print("无效选择")
        
        print()


if __name__ == "__main__":
    main()
