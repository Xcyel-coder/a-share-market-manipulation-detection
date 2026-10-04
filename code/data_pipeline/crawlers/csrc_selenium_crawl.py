"""
证监会行政处罚决定书爬虫 - Selenium版

使用本地Chrome浏览器处理JS动态渲染页面。
策略：从列表页翻页发现所有处罚文书URL → 逐条采集详情页正文 → 筛选操纵类
"""
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

PENALTY_LIST_URL = "http://www.csrc.gov.cn/csrc/c101928/zfxxgk_zdgk.shtml"


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


class CSRCSeleniumCrawler:
    """基于Selenium的证监会处罚决定书爬虫"""

    def __init__(self, headless: bool = True):
        raw_path = config.paths_config.get("raw_data", "data/raw")
        self.output_dir = Path(raw_path) / "csrc_penalties"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = 2
        self.existing_urls = self._load_existing_urls()
        self.driver = self._init_driver(headless)
        log.info(f"Selenium爬虫初始化完成，已有 {len(self.existing_urls)} 个文档")

    def _init_driver(self, headless: bool) -> webdriver.Chrome:
        opts = Options()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36")
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(30)
        return driver

    def _load_existing_urls(self) -> set:
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

    def _detect_keywords(self, text: str) -> List[str]:
        return [kw for kw in MANIPULATION_KEYWORDS if kw in text]

    def _extract_doc_id(self, title: str, content: str, url: str) -> str:
        for text in [title, content[:1000]]:
            for pattern in [r"〔(\d{4})〕(\d+)号", r"\[(\d{4})\](\d+)号"]:
                m = re.search(pattern, text)
                if m:
                    return f"CSRC-{m.group(1)}-{m.group(2).zfill(3)}"
        m = re.search(r"/c(\d+)/content", url)
        if m:
            return f"CSRC-URL-{m.group(1)}"
        return f"CSRC-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _extract_publish_date(self, content: str) -> str:
        for pattern in [r"(\d{4})年(\d{1,2})月(\d{1,2})日", r"(\d{4})-(\d{2})-(\d{2})"]:
            m = re.search(pattern, content[-300:] if content else "")
            if m:
                g = m.groups()
                return f"{g[0]}-{g[1].zfill(2)}-{g[2].zfill(2)}"
        return ""

    def _classify(self, keywords: List[str], content: str) -> str:
        manip_kws = {"操纵", "对倒", "拉抬", "打压", "连续交易", "虚假申报", "洗售", "连续买卖", "集中资金优势", "持股优势", "账户组"}
        if any(kw in manip_kws for kw in keywords):
            return "操纵证券市场"
        if "内幕" in content:
            return "内幕交易"
        if "信息披露" in content or "虚假陈述" in content:
            return "信息披露违规"
        return "其他证券违法"

    def _save_doc(self, doc: PenaltyDocument) -> bool:
        year = doc.publish_date[:4] if doc.publish_date and doc.publish_date[:4].isdigit() else "unknown"
        year_dir = self.output_dir / year
        year_dir.mkdir(exist_ok=True)
        path = year_dir / f"{doc.doc_id}.json"
        if path.exists():
            return False
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(doc), f, ensure_ascii=False, indent=2)
        return True

    def discover_urls_from_list(self, max_pages: int = 50) -> List[Dict]:
        """从列表页翻页发现处罚文书URL（Selenium处理JS渲染）"""
        log.info(f"=== 从列表页发现URL（最多 {max_pages} 页）===")
        all_links = []

        try:
            self.driver.get(PENALTY_LIST_URL)
            time.sleep(3)

            for page_num in range(1, max_pages + 1):
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "a"))
                    )
                except Exception:
                    pass

                soup = BeautifulSoup(self.driver.page_source, "lxml")
                links = soup.find_all("a", href=True)

                page_count = 0
                for link in links:
                    href = link.get("href", "")
                    title = link.get_text(strip=True)
                    if "/content.shtml" in href and len(title) > 5:
                        if not href.startswith("http"):
                            href = f"http://www.csrc.gov.cn{href}"
                        if href not in self.existing_urls:
                            all_links.append({"url": href, "title": title})
                            page_count += 1

                log.info(f"第 {page_num} 页: 发现 {page_count} 个新链接")

                # 翻到下一页
                try:
                    next_btns = self.driver.find_elements(By.CSS_SELECTOR, "a.next, a[title='下一页'], .page-next a")
                    if not next_btns:
                        # 尝试通用翻页按钮
                        all_page_links = self.driver.find_elements(By.CSS_SELECTOR, ".page a, .pagination a")
                        next_btns = [a for a in all_page_links if "下一页" in a.text or ">" == a.text.strip()]

                    if next_btns:
                        next_btns[0].click()
                        time.sleep(self.delay)
                    else:
                        log.info(f"第 {page_num} 页后无更多页面")
                        break
                except Exception as e:
                    log.warning(f"翻页失败: {e}")
                    break

        except Exception as e:
            log.error(f"列表页访问失败: {e}")

        # 去重
        seen = set()
        unique = []
        for link in all_links:
            if link["url"] not in seen:
                seen.add(link["url"])
                unique.append(link)

        log.info(f"共发现 {len(unique)} 个新URL")
        return unique

    def crawl_detail_page(self, url: str, title_hint: str = "") -> Optional[PenaltyDocument]:
        """采集单个详情页（Selenium渲染后提取正文）"""
        if url in self.existing_urls:
            return None

        try:
            self.driver.get(url)
            time.sleep(2)

            soup = BeautifulSoup(self.driver.page_source, "lxml")

            # 提取标题
            title = ""
            h2 = soup.select_one("h2") or soup.select_one("h1")
            if h2:
                title = h2.get_text(strip=True)
            if not title:
                title = title_hint

            # 提取正文
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

            if len(content) < 500:
                paragraphs = soup.select("p")
                if len(paragraphs) > 5:
                    content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

            if len(content) < 300:
                return None

            keywords = self._detect_keywords(title + "\n" + content)

            doc = PenaltyDocument(
                doc_id=self._extract_doc_id(title, content, url),
                title=title,
                url=url,
                publish_date=self._extract_publish_date(content),
                content=content,
                case_type=self._classify(keywords, content),
                keywords_matched=keywords,
                crawl_time=datetime.now().isoformat()
            )

            if self._save_doc(doc):
                self.existing_urls.add(url)
                return doc

        except Exception as e:
            log.warning(f"采集失败 {url}: {e}")

        return None

    def update_index(self):
        """更新索引文件"""
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
        with open(self.output_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        log.info(f"索引更新: {len(all_docs)} 个文档，操纵类 {manipulation_count}")

    def run(self, max_list_pages: int = 50, crawl_all: bool = True):
        """
        执行完整采集

        Args:
            max_list_pages: 列表页最大翻页数
            crawl_all: True=采集所有类型, False=仅采集操纵类（需标题含关键词）
        """
        log.info("=" * 60)
        log.info("证监会处罚决定书 Selenium 采集")
        log.info("=" * 60)

        try:
            # 阶段1：从列表页发现URL
            discovered = self.discover_urls_from_list(max_list_pages)

            # 阶段2：逐条采集详情页
            log.info(f"=== 采集详情页（{len(discovered)} 个）===")
            docs = []
            for item in tqdm(discovered, desc="采集详情"):
                doc = self.crawl_detail_page(item["url"], item.get("title", ""))
                if doc:
                    log.info(f"保存: {doc.doc_id} [{doc.case_type}] {doc.title[:40]}")
                    docs.append(doc)
                time.sleep(self.delay)

            self.update_index()

            manip = [d for d in docs if d.case_type == "操纵证券市场"]
            log.info("=" * 60)
            log.info(f"采集完成！新增 {len(docs)} 个文档，其中操纵类 {len(manip)} 个")
            log.info("=" * 60)

        finally:
            self.driver.quit()
            log.info("浏览器已关闭")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="证监会处罚文书Selenium采集")
    parser.add_argument("--max-pages", type=int, default=50, help="列表页最大翻页数")
    parser.add_argument("--no-headless", action="store_true", help="显示浏览器窗口（调试用）")
    args = parser.parse_args()

    crawler = CSRCSeleniumCrawler(headless=not args.no_headless)
    crawler.run(max_list_pages=args.max_pages)


if __name__ == "__main__":
    main()
