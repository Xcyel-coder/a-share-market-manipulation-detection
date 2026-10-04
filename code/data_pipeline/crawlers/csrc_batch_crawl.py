#!/usr/bin/env python
"""
证监会行政处罚决定书批量采集脚本

策略：使用已知的URL列表 + WebFetch直接获取正文内容。
证监会官网详情页（content.shtml）可通过HTTP直接访问，不需要JS渲染。
"""
import json
import re
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config

MANIPULATION_KEYWORDS = [
    "操纵", "对倒", "对敲", "拉抬", "打压",
    "虚假申报", "撤单", "洗售", "相对委托",
    "尾市", "连续交易", "约定交易", "连续买卖",
    "集中资金优势", "持股优势", "账户组",
]

# ============================================================
# 已知的证监会处罚决定书URL列表
# 来源：证监会官网 行政处罚栏目
# 路径：政务信息 → 行政执法 → 行政处罚
# ============================================================
KNOWN_PENALTY_URLS = [
    # 2024年操纵类
    ("http://www.csrc.gov.cn/csrc/c101928/c7500884/content.shtml", "朱要文赵杨王立华操纵*ST金洲 [2024]58号"),
    ("http://www.csrc.gov.cn/csrc/c101928/c7520315/content.shtml", "刘洪涛娄阁操纵腾茂科技 [2024]120号"),
    ("http://www.csrc.gov.cn/csrc/c101928/c7532746/content.shtml", "汤秀清肖泳林操纵昊志机电 [2024]141号"),
    ("http://www.csrc.gov.cn/csrc/c101928/c7471690/content.shtml", "王斌操纵证券 2024"),
    # 2022年
    ("http://www.csrc.gov.cn/csrc/c101928/c6932470/content.shtml", "王宝元操纵8只股票 [2022]64号"),
    ("https://www.csrc.gov.cn/csrc/c101928/c1042623/content.shtml", "刘文金操纵证券"),
    # 地方监管局
    ("https://www.csrc.gov.cn/guangxi/c104662/c1664579/content.shtml", "蒋卫强 广西监管局 [2021]4号"),
    ("https://www.csrc.gov.cn/tianjin/c103608/c7548029/content.shtml", "徐阳操纵多只股票 天津监管局 [2025]1号"),
]

# 证监会处罚列表页（按时间倒序排列，每页约20条）
PENALTY_LIST_PAGES = [
    "http://www.csrc.gov.cn/csrc/c101928/zfxxgk_zdgk.shtml",
]
# 根据URL规律生成更多列表页
for i in range(2, 51):
    PENALTY_LIST_PAGES.append(f"http://www.csrc.gov.cn/csrc/c101928/zfxxgk_zdgk_{i}.shtml")


@dataclass
class PenaltyDocument:
    doc_id: str
    title: str
    url: str
    publish_date: str
    content: str
    case_type: str
    keywords_matched: List[str]
    crawl_time: str


class CSRCBatchCrawler:
    """证监会处罚决定书批量采集器"""
    
    def __init__(self):
        raw_path = config.paths_config.get("raw_data", "data/raw")
        self.output_dir = Path(raw_path) / "csrc_penalties"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        self.delay = 2
        self.existing_urls = self._load_existing_urls()
        log.info(f"批量采集器初始化，已有 {len(self.existing_urls)} 个文档")
    
    def _load_existing_urls(self) -> set:
        """加载已采集的URL避免重复"""
        urls = set()
        for f in self.output_dir.rglob("*.json"):
            if f.name == "index.json":
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("url"):
                    urls.add(data["url"])
            except Exception:
                pass
        return urls
    
    def _fetch_page(self, url: str, retries: int = 3) -> Optional[str]:
        """获取页面HTML"""
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    if resp.encoding == 'ISO-8859-1':
                        resp.encoding = resp.apparent_encoding
                    return resp.text
                log.warning(f"HTTP {resp.status_code}: {url}")
            except Exception as e:
                log.warning(f"请求失败 ({attempt+1}/{retries}): {url}, {e}")
                time.sleep(self.delay * (attempt + 1))
        return None
    
    def _extract_content_from_detail(self, html: str) -> Tuple[str, str, str]:
        """
        从详情页提取正文、标题、日期
        Returns: (content, title, publish_date)
        """
        soup = BeautifulSoup(html, "lxml")
        
        # 提取标题
        title = ""
        title_el = soup.select_one("meta[name='ArticleTitle']")
        if title_el:
            title = title_el.get("content", "")
        if not title:
            h2 = soup.select_one("h2") or soup.select_one("h1")
            if h2:
                title = h2.get_text(strip=True)
        
        # 提取正文：证监会详情页使用多种容器
        content = ""
        for selector in [
            "#ContentRegion", "div.TRS_Editor", "div.detail_content",
            "div.content_detail", "div.article-content", "article",
            "div.xxgk-detail-content", "div.content"
        ]:
            el = soup.select_one(selector)
            if el:
                for tag in el.select("script, style"):
                    tag.decompose()
                content = el.get_text(separator="\n", strip=True)
                if len(content) > 500:
                    break
        
        # 如果上述选择器都没拿到足够内容，尝试拿所有段落
        if len(content) < 500:
            paragraphs = soup.select("p")
            if len(paragraphs) > 5:
                content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        
        # 提取日期
        publish_date = ""
        date_el = soup.select_one("meta[name='PubDate']")
        if date_el:
            publish_date = date_el.get("content", "")
        if not publish_date:
            date_match = re.search(r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})", content[-200:] if content else "")
            if date_match:
                g = date_match.groups()
                publish_date = f"{g[0]}-{g[1].zfill(2)}-{g[2].zfill(2)}"
        
        return content, title, publish_date
    
    def _detect_manipulation(self, text: str) -> List[str]:
        """检测操纵类关键词"""
        return [kw for kw in MANIPULATION_KEYWORDS if kw in text]
    
    def _extract_doc_id(self, title: str, content: str, url: str) -> str:
        """提取文书编号"""
        for text in [title, content[:1000]]:
            for pattern in [r"〔(\d{4})〕(\d+)号", r"\[(\d{4})\](\d+)号"]:
                m = re.search(pattern, text)
                if m:
                    return f"CSRC-{m.group(1)}-{m.group(2).zfill(3)}"
        # 从URL提取
        m = re.search(r"/c(\d+)/content", url)
        if m:
            return f"CSRC-URL-{m.group(1)}"
        return f"CSRC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _save_doc(self, doc: PenaltyDocument) -> bool:
        """保存文档"""
        year = doc.publish_date[:4] if doc.publish_date and doc.publish_date[:4].isdigit() else "unknown"
        year_dir = self.output_dir / year
        year_dir.mkdir(exist_ok=True)
        
        path = year_dir / f"{doc.doc_id}.json"
        if path.exists():
            log.info(f"已存在，跳过: {doc.doc_id}")
            return False
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(doc), f, ensure_ascii=False, indent=2)
        return True
    
    def crawl_single_url(self, url: str, hint: str = "") -> Optional[PenaltyDocument]:
        """采集单条URL"""
        if url in self.existing_urls:
            log.info(f"已采集，跳过: {hint or url}")
            return None
        
        html = self._fetch_page(url)
        if not html:
            return None
        
        content, title, pub_date = self._extract_content_from_detail(html)
        if not content or len(content) < 300:
            log.warning(f"内容过短: {url}")
            return None
        
        keywords = self._detect_manipulation(title + "\n" + content)
        
        case_type = "操纵证券市场" if keywords else "其他证券违法"
        
        doc = PenaltyDocument(
            doc_id=self._extract_doc_id(title, content, url),
            title=title or hint,
            url=url,
            publish_date=pub_date or datetime.now().strftime("%Y-%m-%d"),
            content=content,
            case_type=case_type,
            keywords_matched=keywords,
            crawl_time=datetime.now().isoformat()
        )
        
        if self._save_doc(doc):
            self.existing_urls.add(url)
            log.info(f"保存成功: {doc.doc_id} [{doc.case_type}] {title[:40]}")
            return doc
        return None
    
    def crawl_known_urls(self) -> List[PenaltyDocument]:
        """采集已知URL列表"""
        log.info(f"=== 阶段1: 采集已知URL ({len(KNOWN_PENALTY_URLS)} 个) ===")
        docs = []
        for url, hint in tqdm(KNOWN_PENALTY_URLS, desc="已知URL"):
            doc = self.crawl_single_url(url, hint)
            if doc:
                docs.append(doc)
            time.sleep(self.delay)
        log.info(f"已知URL采集完成: {len(docs)} 个新文档")
        return docs
    
    def discover_from_list_pages(self, max_pages: int = 50) -> List[str]:
        """
        从证监会列表页发现更多URL
        注意：列表页可能需要JS渲染，但部分内容可直接通过HTML获取
        """
        log.info(f"=== 阶段2: 从列表页发现URL (最多 {max_pages} 页) ===")
        discovered = []
        
        for i, list_url in enumerate(PENALTY_LIST_PAGES[:max_pages]):
            html = self._fetch_page(list_url)
            if not html:
                log.info(f"列表页 {i+1} 不可用，尝试下一页")
                continue
            
            soup = BeautifulSoup(html, "lxml")
            
            # 查找所有链接
            links = soup.find_all("a", href=True)
            for link in links:
                href = link.get("href", "")
                text = link.get_text(strip=True)
                
                # 筛选处罚决定书详情页链接
                if "/content.shtml" in href and ("处罚" in text or "决定书" in text):
                    if not href.startswith("http"):
                        href = f"http://www.csrc.gov.cn{href}"
                    if href not in self.existing_urls and href not in discovered:
                        discovered.append(href)
            
            log.info(f"列表页 {i+1}: 发现 {len(links)} 个链接")
            time.sleep(self.delay)
        
        log.info(f"列表页发现: {len(discovered)} 个新URL")
        return discovered
    
    def crawl_discovered_urls(self, urls: List[str], only_manipulation: bool = False) -> List[PenaltyDocument]:
        """采集发现的URL"""
        log.info(f"=== 阶段3: 采集发现的URL ({len(urls)} 个) ===")
        docs = []
        
        for url in tqdm(urls, desc="采集详情"):
            doc = self.crawl_single_url(url)
            if doc:
                if only_manipulation and not doc.keywords_matched:
                    continue
                docs.append(doc)
            time.sleep(self.delay)
        
        log.info(f"详情采集完成: {len(docs)} 个新文档")
        return docs
    
    def update_index(self):
        """更新索引"""
        all_docs = []
        for f in self.output_dir.rglob("*.json"):
            if f.name == "index.json":
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                all_docs.append({
                    "doc_id": data.get("doc_id"),
                    "title": data.get("title", "")[:60],
                    "case_type": data.get("case_type"),
                    "publish_date": data.get("publish_date"),
                    "keywords": data.get("keywords_matched", []),
                    "url": data.get("url"),
                })
            except Exception:
                pass
        
        manipulation_count = sum(1 for d in all_docs if d.get("case_type") == "操纵证券市场")
        
        index = {
            "total": len(all_docs),
            "manipulation_cases": manipulation_count,
            "other_cases": len(all_docs) - manipulation_count,
            "update_time": datetime.now().isoformat(),
            "files": all_docs
        }
        
        index_path = self.output_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        log.info(f"索引更新: 共 {len(all_docs)} 个文档，操纵类 {manipulation_count} 个")
    
    def run(self, max_list_pages: int = 30):
        """执行完整采集"""
        log.info("=" * 60)
        log.info("证监会处罚决定书批量采集")
        log.info("=" * 60)
        
        all_docs = []
        
        # 阶段1：采集已知URL
        docs1 = self.crawl_known_urls()
        all_docs.extend(docs1)
        
        # 阶段2：从列表页发现更多URL
        discovered = self.discover_from_list_pages(max_list_pages)
        
        # 阶段3：采集发现的URL
        if discovered:
            docs2 = self.crawl_discovered_urls(discovered)
            all_docs.extend(docs2)
        
        # 更新索引
        self.update_index()
        
        log.info("=" * 60)
        log.info(f"采集完成！新增 {len(all_docs)} 个文档")
        manipulation = [d for d in all_docs if d.keywords_matched]
        log.info(f"其中操纵类: {len(manipulation)} 个")
        log.info("=" * 60)
        
        return all_docs


def main():
    crawler = CSRCBatchCrawler()
    crawler.run(max_list_pages=30)


if __name__ == "__main__":
    main()
