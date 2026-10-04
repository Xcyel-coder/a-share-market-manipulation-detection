"""
证监会行政处罚决定书爬虫

爬取中国证监会官网的行政处罚决定书，重点关注操纵类案件。
"""
import re
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config


@dataclass
class PenaltyDocument:
    """处罚决定书数据结构"""
    doc_id: str                    # 文书编号
    title: str                     # 标题
    url: str                       # 原文链接
    publish_date: str              # 发布日期
    content: str                   # 正文内容
    case_type: str                 # 案件类型（操纵、内幕交易等）
    keywords_matched: List[str]    # 匹配的关键词
    crawl_time: str                # 爬取时间


class CSRCCrawler:
    """证监会处罚决定书爬虫"""
    
    def __init__(self):
        """初始化爬虫"""
        self.config = config.csrc_config
        self.base_url = self.config.get("base_url", "http://www.csrc.gov.cn")
        self.keywords = self.config.get("manipulation_keywords", [])
        self.delay = self.config.get("request_delay", 2)
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        
        # 请求头
        self.ua = UserAgent()
        self.session = requests.Session()
        
        # 数据存储
        self.output_dir = Path(config.paths_config.get("raw_data", "data/raw")) / "csrc_penalties"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"证监会爬虫初始化完成，输出目录: {self.output_dir}")
    
    def _get_headers(self) -> Dict[str, str]:
        """生成请求头"""
        return {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Referer": self.base_url
        }
    
    def _request_with_retry(self, url: str, method: str = "GET", **kwargs) -> Optional[requests.Response]:
        """
        带重试的请求
        
        Args:
            url: 请求URL
            method: 请求方法
            **kwargs: 其他请求参数
            
        Returns:
            响应对象或None
        """
        for attempt in range(self.max_retries):
            try:
                kwargs.setdefault("headers", self._get_headers())
                kwargs.setdefault("timeout", self.timeout)
                
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                
                # 检测编码
                if response.encoding == 'ISO-8859-1':
                    response.encoding = response.apparent_encoding
                
                return response
                
            except requests.RequestException as e:
                log.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {url}, 错误: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
        
        log.error(f"请求最终失败: {url}")
        return None
    
    def _is_manipulation_case(self, title: str, content: str = "") -> Tuple[bool, List[str]]:
        """
        判断是否为操纵类案件
        
        Args:
            title: 标题
            content: 正文内容
            
        Returns:
            (是否为操纵案件, 匹配的关键词列表)
        """
        text = title + " " + content
        matched = [kw for kw in self.keywords if kw in text]
        return len(matched) > 0, matched
    
    def _extract_doc_id(self, title: str, url: str) -> str:
        """
        提取文书编号
        
        Args:
            title: 标题
            url: URL
            
        Returns:
            文书编号
        """
        # 尝试从标题提取，如 "[2023]1号"
        pattern = r'\[(\d{4})\](\d+)号'
        match = re.search(pattern, title)
        if match:
            return f"CSRC-{match.group(1)}-{match.group(2)}"
        
        # 使用URL中的ID
        url_pattern = r'/(\d+)\.s?html?$'
        match = re.search(url_pattern, url)
        if match:
            return f"CSRC-URL-{match.group(1)}"
        
        # 使用时间戳
        return f"CSRC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def get_penalty_list_page(self, page: int = 1) -> List[Dict]:
        """
        获取处罚决定书列表页
        
        Args:
            page: 页码
            
        Returns:
            文档列表
        """
        # 证监会处罚公开列表页URL（示例，实际URL可能需要调整）
        if page == 1:
            list_url = f"{self.base_url}/csrc/c101928/zfxxgk_zdgk.shtml"
        else:
            list_url = f"{self.base_url}/csrc/c101928/zfxxgk_zdgk_{page}.shtml"
        
        log.info(f"获取列表页: {list_url}")
        
        response = self._request_with_retry(list_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'lxml')
        documents = []
        
        # 解析列表（根据实际网页结构调整选择器）
        items = soup.select('ul.list_con li, div.list_content li, table.list_table tr')
        
        for item in items:
            try:
                # 提取链接和标题
                link = item.select_one('a')
                if not link:
                    continue
                
                title = link.get_text(strip=True)
                href = link.get('href', '')
                
                # 构建完整URL
                if href and not href.startswith('http'):
                    href = urljoin(self.base_url, href)
                
                # 提取日期
                date_elem = item.select_one('span.date, td.date, span.time')
                publish_date = date_elem.get_text(strip=True) if date_elem else ""
                
                # 初步筛选：标题包含关键词
                is_manipulation, keywords = self._is_manipulation_case(title)
                
                if is_manipulation:
                    documents.append({
                        "title": title,
                        "url": href,
                        "publish_date": publish_date,
                        "keywords_matched": keywords
                    })
                    log.debug(f"发现操纵类案件: {title}")
                    
            except Exception as e:
                log.warning(f"解析列表项失败: {e}")
                continue
        
        return documents
    
    def get_document_content(self, url: str) -> Optional[str]:
        """
        获取文档正文内容
        
        Args:
            url: 文档URL
            
        Returns:
            正文内容
        """
        response = self._request_with_retry(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 尝试多种选择器提取正文（根据实际网页结构）
        content_selectors = [
            'div.TRS_Editor',
            'div.content',
            'div.article_content',
            'div.detail_content',
            'article',
            'div#content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # 移除脚本和样式
                for tag in content_elem.select('script, style'):
                    tag.decompose()
                
                content = content_elem.get_text(separator='\n', strip=True)
                if len(content) > 100:  # 确保有足够内容
                    return content
        
        # 兜底：获取body内容
        body = soup.select_one('body')
        if body:
            for tag in body.select('script, style, header, footer, nav'):
                tag.decompose()
            return body.get_text(separator='\n', strip=True)
        
        return None
    
    def crawl_single_document(self, doc_info: Dict) -> Optional[PenaltyDocument]:
        """
        爬取单个文档
        
        Args:
            doc_info: 文档基本信息
            
        Returns:
            处罚决定书对象
        """
        url = doc_info.get("url", "")
        title = doc_info.get("title", "")
        
        log.info(f"爬取文档: {title}")
        
        # 获取正文
        content = self.get_document_content(url)
        if not content:
            log.warning(f"获取正文失败: {url}")
            return None
        
        # 二次验证是否为操纵类案件
        is_manipulation, keywords = self._is_manipulation_case(title, content)
        if not is_manipulation:
            log.debug(f"非操纵类案件（二次验证）: {title}")
            return None
        
        # 构建文档对象
        doc = PenaltyDocument(
            doc_id=self._extract_doc_id(title, url),
            title=title,
            url=url,
            publish_date=doc_info.get("publish_date", ""),
            content=content,
            case_type="操纵证券市场",
            keywords_matched=keywords,
            crawl_time=datetime.now().isoformat()
        )
        
        return doc
    
    def save_document(self, doc: PenaltyDocument) -> str:
        """
        保存文档
        
        Args:
            doc: 处罚决定书对象
            
        Returns:
            保存路径
        """
        # 按年份组织目录
        year_match = re.search(r'20\d{2}', doc.publish_date or doc.crawl_time)
        year = year_match.group() if year_match else "unknown"
        
        year_dir = self.output_dir / year
        year_dir.mkdir(exist_ok=True)
        
        # 保存JSON
        file_path = year_dir / f"{doc.doc_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(doc), f, ensure_ascii=False, indent=2)
        
        log.info(f"文档已保存: {file_path}")
        return str(file_path)
    
    def crawl(self, max_pages: int = 50, save_interval: int = 10) -> List[str]:
        """
        执行爬取任务
        
        Args:
            max_pages: 最大爬取页数
            save_interval: 保存间隔（每隔多少个文档保存一次索引）
            
        Returns:
            已保存的文件路径列表
        """
        log.info(f"开始爬取证监会处罚决定书，最大页数: {max_pages}")
        
        saved_files = []
        all_docs_info = []
        
        # 第一阶段：获取列表
        log.info("=== 阶段1: 获取文档列表 ===")
        for page in tqdm(range(1, max_pages + 1), desc="获取列表"):
            docs = self.get_penalty_list_page(page)
            if not docs:
                log.info(f"第 {page} 页无数据，停止获取列表")
                break
            
            all_docs_info.extend(docs)
            time.sleep(self.delay)
        
        log.info(f"共发现 {len(all_docs_info)} 个潜在操纵类案件")
        
        # 去重
        seen_urls = set()
        unique_docs = []
        for doc in all_docs_info:
            if doc['url'] not in seen_urls:
                seen_urls.add(doc['url'])
                unique_docs.append(doc)
        
        log.info(f"去重后: {len(unique_docs)} 个案件")
        
        # 第二阶段：爬取详情
        log.info("=== 阶段2: 爬取文档详情 ===")
        for i, doc_info in enumerate(tqdm(unique_docs, desc="爬取详情")):
            try:
                doc = self.crawl_single_document(doc_info)
                if doc:
                    file_path = self.save_document(doc)
                    saved_files.append(file_path)
                
                # 控制请求频率
                time.sleep(self.delay)
                
                # 定期保存索引
                if (i + 1) % save_interval == 0:
                    self._save_index(saved_files)
                    
            except Exception as e:
                log.error(f"处理文档失败: {doc_info.get('title', 'unknown')}, 错误: {e}")
                continue
        
        # 最终保存索引
        self._save_index(saved_files)
        
        log.info(f"爬取完成，共保存 {len(saved_files)} 个文档")
        return saved_files
    
    def _save_index(self, saved_files: List[str]):
        """保存文档索引"""
        index_path = self.output_dir / "index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total": len(saved_files),
                "update_time": datetime.now().isoformat(),
                "files": saved_files
            }, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    crawler = CSRCCrawler()
    
    # 执行爬取
    saved_files = crawler.crawl(max_pages=100)
    
    print(f"\n爬取完成！共保存 {len(saved_files)} 个操纵类处罚决定书")
    print(f"数据保存在: {crawler.output_dir}")


if __name__ == "__main__":
    main()

