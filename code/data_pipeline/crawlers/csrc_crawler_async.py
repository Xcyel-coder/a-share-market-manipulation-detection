"""
证监会行政处罚决定书爬虫 - 异步版本

使用Playwright处理JavaScript动态加载的页面。
"""
import re
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config


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


class CSRCAsyncCrawler:
    """证监会处罚决定书异步爬虫"""
    
    # 操纵类关键词
    MANIPULATION_KEYWORDS = [
        "操纵", "对倒", "对敲", "拉抬", "打压",
        "虚假申报", "撤单", "洗售", "相对委托",
        "尾市", "开盘", "连续交易", "约定交易"
    ]
    
    # 处罚决定书列表页面URL
    LIST_URLS = [
        # 证监会本级行政处罚
        "http://www.csrc.gov.cn/csrc/c101928/zfxxgk_zdgk.shtml",
        # 备用URL - 旧版列表页
        "http://www.csrc.gov.cn/csrc/c100230/common_list.shtml",
    ]
    
    def __init__(self):
        """初始化爬虫"""
        self.config = config.csrc_config if hasattr(config, 'csrc_config') else {}
        self.keywords = self.MANIPULATION_KEYWORDS
        self.delay = self.config.get("request_delay", 2)
        
        # 数据存储目录
        raw_data_path = config.paths_config.get("raw_data", "data/raw") if hasattr(config, 'paths_config') else "data/raw"
        self.output_dir = Path(raw_data_path) / "csrc_penalties"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.browser: Optional[Browser] = None
        self.documents: List[PenaltyDocument] = []
        
        log.info(f"证监会异步爬虫初始化完成，输出目录: {self.output_dir}")
    
    async def init_browser(self, headless: bool = True):
        """初始化浏览器"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=headless)
        log.info("浏览器已启动")
    
    async def close_browser(self):
        """关闭浏览器"""
        if self.browser:
            await self.browser.close()
            log.info("浏览器已关闭")
    
    def _contains_manipulation_keywords(self, text: str) -> List[str]:
        """检查文本是否包含操纵类关键词"""
        matched = []
        for keyword in self.keywords:
            if keyword in text:
                matched.append(keyword)
        return matched
    
    async def get_penalty_list(self, page: Page, max_pages: int = 10) -> List[Dict]:
        """
        获取处罚决定书列表
        
        Args:
            page: Playwright页面对象
            max_pages: 最大爬取页数
            
        Returns:
            文档链接列表
        """
        all_links = []
        
        # 访问列表页
        for list_url in self.LIST_URLS:
            try:
                log.info(f"访问列表页: {list_url}")
                await page.goto(list_url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(2)  # 等待JavaScript渲染
                
                # 获取页面内容
                content = await page.content()
                
                # 检查是否有列表数据
                if "暂无数据" not in content and "common-list" in content:
                    break
            except Exception as e:
                log.warning(f"访问 {list_url} 失败: {e}")
                continue
        
        # 爬取多页
        for page_num in range(1, max_pages + 1):
            try:
                log.info(f"正在爬取第 {page_num} 页...")
                
                # 获取当前页面内容
                content = await page.content()
                soup = BeautifulSoup(content, "lxml")
                
                # 查找列表项
                list_items = soup.select("ul.common-list li a") or \
                            soup.select("div.list-content li a") or \
                            soup.select(".zfxx-list li a") or \
                            soup.select("li.clearfix a")
                
                if not list_items:
                    log.warning(f"第 {page_num} 页未找到列表项")
                    # 尝试其他选择器
                    list_items = soup.find_all("a", href=re.compile(r"/csrc/c\d+/\w+\.shtml"))
                
                for item in list_items:
                    title = item.get_text(strip=True)
                    href = item.get("href", "")
                    
                    # 只关注包含操纵类关键词的处罚决定
                    if not href or not title:
                        continue
                    
                    keywords_matched = self._contains_manipulation_keywords(title)
                    
                    # 也检查是否是行政处罚决定书
                    if "行政处罚" in title or "处罚决定" in title or keywords_matched:
                        # 构建完整URL
                        if not href.startswith("http"):
                            href = f"http://www.csrc.gov.cn{href}"
                        
                        all_links.append({
                            "title": title,
                            "url": href,
                            "keywords_matched": keywords_matched
                        })
                
                log.info(f"第 {page_num} 页找到 {len(list_items)} 个链接")
                
                # 点击下一页
                next_btn = await page.query_selector("a.next") or \
                           await page.query_selector("a:has-text('下一页')") or \
                           await page.query_selector(".page-next")
                
                if next_btn:
                    await next_btn.click()
                    await asyncio.sleep(self.delay)
                else:
                    log.info("没有更多页面")
                    break
                    
            except Exception as e:
                log.error(f"爬取第 {page_num} 页失败: {e}")
                break
        
        # 去重
        seen_urls = set()
        unique_links = []
        for link in all_links:
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                unique_links.append(link)
        
        log.info(f"共获取 {len(unique_links)} 个唯一链接")
        return unique_links
    
    async def get_document_content(self, page: Page, url: str) -> Optional[str]:
        """
        获取文档正文内容
        
        Args:
            page: Playwright页面对象
            url: 文档URL
            
        Returns:
            文档正文
        """
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(1)
            
            content = await page.content()
            soup = BeautifulSoup(content, "lxml")
            
            # 尝试多种正文选择器
            content_selectors = [
                ".TRS_Editor",
                ".article-content",
                ".content",
                "#Zoom",
                ".TRS_PreAppend",
                "div.xxgk-detail-content"
            ]
            
            for selector in content_selectors:
                article = soup.select_one(selector)
                if article:
                    # 获取纯文本
                    text = article.get_text(separator="\n", strip=True)
                    if len(text) > 100:  # 确保有足够内容
                        return text
            
            # 备选方案：获取所有段落
            paragraphs = soup.select("p")
            if paragraphs:
                text = "\n".join([p.get_text(strip=True) for p in paragraphs])
                if len(text) > 100:
                    return text
            
            log.warning(f"未能提取正文: {url}")
            return None
            
        except Exception as e:
            log.error(f"获取文档内容失败 {url}: {e}")
            return None
    
    def _extract_doc_id(self, title: str, url: str) -> str:
        """从标题或URL提取文书编号"""
        # 尝试从标题提取
        patterns = [
            r"〔(\d{4})〕(\d+)号",
            r"\[(\d{4})\](\d+)号",
            r"（(\d{4})）(\d+)号"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                year, num = match.groups()
                return f"CSRC-{year}-{num.zfill(3)}"
        
        # 从URL提取
        url_match = re.search(r"/(\w+)\.shtml", url)
        if url_match:
            return f"CSRC-{url_match.group(1)}"
        
        # 使用时间戳
        return f"CSRC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _extract_publish_date(self, content: str, soup: BeautifulSoup = None) -> str:
        """提取发布日期"""
        patterns = [
            r"(\d{4})年(\d{1,2})月(\d{1,2})日",
            r"(\d{4})-(\d{2})-(\d{2})",
            r"(\d{4})/(\d{2})/(\d{2})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                groups = match.groups()
                return f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
        
        return datetime.now().strftime("%Y-%m-%d")
    
    def _determine_case_type(self, keywords: List[str], content: str) -> str:
        """判断案件类型"""
        if any(kw in keywords for kw in ["操纵", "对倒", "拉抬", "打压"]):
            return "操纵证券市场"
        elif "内幕" in content:
            return "内幕交易"
        elif "信息披露" in content or "虚假陈述" in content:
            return "信息披露违规"
        else:
            return "其他证券违法"
    
    def save_document(self, doc: PenaltyDocument):
        """保存单个文档"""
        # 按年份组织
        year = doc.publish_date[:4] if doc.publish_date else datetime.now().strftime("%Y")
        year_dir = self.output_dir / year
        year_dir.mkdir(exist_ok=True)
        
        # 保存为JSON
        file_path = year_dir / f"{doc.doc_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(doc), f, ensure_ascii=False, indent=2)
        
        log.info(f"文档已保存: {file_path}")
    
    async def crawl(self, max_pages: int = 10, max_docs: int = 100):
        """
        执行爬取
        
        Args:
            max_pages: 最大爬取页数
            max_docs: 最大爬取文档数
        """
        await self.init_browser(headless=True)
        
        try:
            page = await self.browser.new_page()
            
            # 设置User-Agent
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            
            log.info("=== 阶段1: 获取文档列表 ===")
            doc_links = await self.get_penalty_list(page, max_pages)
            
            # 筛选操纵类案件
            manipulation_links = [
                link for link in doc_links 
                if link.get("keywords_matched") or "操纵" in link.get("title", "")
            ]
            
            log.info(f"筛选出 {len(manipulation_links)} 个潜在操纵类案件")
            
            log.info("=== 阶段2: 爬取文档详情 ===")
            saved_count = 0
            
            for link in tqdm(manipulation_links[:max_docs], desc="爬取详情"):
                try:
                    content = await self.get_document_content(page, link["url"])
                    
                    if not content:
                        continue
                    
                    # 再次验证是否为操纵类案件
                    keywords_matched = self._contains_manipulation_keywords(content)
                    if not keywords_matched:
                        continue
                    
                    # 构建文档对象
                    doc = PenaltyDocument(
                        doc_id=self._extract_doc_id(link["title"], link["url"]),
                        title=link["title"],
                        url=link["url"],
                        publish_date=self._extract_publish_date(content),
                        content=content,
                        case_type=self._determine_case_type(keywords_matched, content),
                        keywords_matched=keywords_matched,
                        crawl_time=datetime.now().isoformat()
                    )
                    
                    self.save_document(doc)
                    self.documents.append(doc)
                    saved_count += 1
                    
                    await asyncio.sleep(self.delay)
                    
                except Exception as e:
                    log.error(f"处理文档失败 {link['url']}: {e}")
                    continue
            
            log.info(f"爬取完成，共保存 {saved_count} 个文档")
            return saved_count
            
        finally:
            await self.close_browser()
    
    def save_index(self):
        """保存索引文件"""
        index_data = {
            "total": len(self.documents),
            "update_time": datetime.now().isoformat(),
            "files": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "case_type": doc.case_type,
                    "publish_date": doc.publish_date
                }
                for doc in self.documents
            ]
        }
        
        index_path = self.output_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        log.info(f"索引已保存: {index_path}")


async def main():
    """主函数"""
    crawler = CSRCAsyncCrawler()
    await crawler.crawl(max_pages=5, max_docs=50)
    crawler.save_index()


if __name__ == "__main__":
    asyncio.run(main())

