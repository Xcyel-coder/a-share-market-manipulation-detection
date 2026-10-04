# -*- coding: utf-8 -*-
"""
阶段二+三：定向爬取 CSRC 处罚决定书

策略：
1. 对 B 类案例（有 penalty_id），搜索 CSRC 全站找到对应的处罚决定书 URL
2. 对 C 类案例（无 penalty_id），用股票名搜索
3. 爬取匹配到的详情页并保存 JSON

注意：penalty_id 可能来自地方证监局（沪〔2022〕44号 vs 〔2022〕44号），
因此搜索时同时使用股票名和文号进行匹配。
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

P = lambda *a, **kw: print(*a, **kw, flush=True)

BASE = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文")
GAP_JSON = BASE / "code" / "_gap_analysis_result.json"
CSRC_DIR = BASE / "code" / "data_pipeline" / "data" / "raw" / "csrc_penalties"
RESULT_PATH = BASE / "code" / "_crawl_result.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
DELAY = 2.5

MANIPULATION_KEYWORDS = [
    "操纵", "对倒", "对敲", "拉抬", "打压",
    "虚假申报", "撤单", "洗售", "相对委托",
    "尾市", "连续交易", "约定交易", "连续买卖",
    "集中资金优势", "持股优势", "账户组",
]


def fetch_page(url: str, session: requests.Session, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                if resp.encoding == "ISO-8859-1":
                    resp.encoding = resp.apparent_encoding
                return resp.text
        except Exception as e:
            if attempt == retries - 1:
                P(f"    请求失败: {e}")
            time.sleep(DELAY * (attempt + 1))
    return None


# ─── CSRC Search via DuckDuckGo ───────────────────────────────────────────────

def search_ddg(query: str) -> list[dict]:
    """Search DuckDuckGo for CSRC URLs. Returns list of {url, title}."""
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=8))
        results = []
        for r in raw:
            href = r.get("href", "")
            title = r.get("title", "")
            if "csrc.gov.cn" in href:
                results.append({"url": href, "title": title})
        return results
    except Exception as e:
        P(f"    搜索异常: {e}")
        return []


def search_for_case(stock_name: str, penalty_pattern: str | None) -> list[dict]:
    """Search for a CSRC penalty document for a given case."""
    clean_name = stock_name.replace("ST ", "").replace("*ST ", "").strip()

    queries = []
    if penalty_pattern:
        queries.append(f'site:csrc.gov.cn {penalty_pattern} {clean_name}')
        queries.append(f'site:csrc.gov.cn {penalty_pattern} 操纵')
    queries.append(f'site:csrc.gov.cn {clean_name} 操纵 行政处罚')

    all_results = []
    seen_urls = set()
    for q in queries:
        results = search_ddg(q)
        for r in results:
            if r["url"] not in seen_urls and "content.shtml" in r["url"]:
                seen_urls.add(r["url"])
                all_results.append(r)
        if all_results:
            break
        time.sleep(DELAY)

    return all_results


# ─── CSRC Detail Page Crawl ──────────────────────────────────────────────────

def extract_content(html: str) -> tuple[str, str, str]:
    """Extract (content, title, publish_date) from CSRC detail page."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    title_el = soup.select_one("meta[name='ArticleTitle']")
    if title_el:
        title = title_el.get("content", "")
    if not title:
        h = soup.select_one("h2") or soup.select_one("h1")
        if h:
            title = h.get_text(strip=True)

    content = ""
    for sel in [
        "#ContentRegion", "div.TRS_Editor", "div.detail_content",
        "div.content_detail", "div.article-content", "article",
        "div.xxgk-detail-content", "div.content",
    ]:
        el = soup.select_one(sel)
        if el:
            for tag in el.select("script, style"):
                tag.decompose()
            content = el.get_text(separator="\n", strip=True)
            if len(content) > 500:
                break
    if len(content) < 500:
        paragraphs = soup.select("p")
        if len(paragraphs) > 5:
            content = "\n".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

    pub_date = ""
    date_el = soup.select_one("meta[name='PubDate']")
    if date_el:
        pub_date = date_el.get("content", "")
    if not pub_date and content:
        dm = re.search(r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})", content[-200:])
        if dm:
            g = dm.groups()
            pub_date = f"{g[0]}-{g[1].zfill(2)}-{g[2].zfill(2)}"

    return content, title, pub_date


def extract_doc_id(title: str, content: str, url: str) -> str:
    for text in [title, content[:1500]]:
        for pattern in [r"〔(\d{4})〕(\d+)号", r"\[(\d{4})\](\d+)号"]:
            m = re.search(pattern, text)
            if m:
                return f"CSRC-{m.group(1)}-{int(m.group(2)):03d}"
    m = re.search(r"/c(\d+)/content", url)
    if m:
        return f"CSRC-URL-{m.group(1)}"
    return f"CSRC-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def crawl_and_save(url: str, session: requests.Session, hint: str = "") -> dict | None:
    html = fetch_page(url, session)
    if not html:
        return None

    content, title, pub_date = extract_content(html)
    if not content or len(content) < 300:
        P(f"    内容过短，跳过")
        return None

    keywords = [kw for kw in MANIPULATION_KEYWORDS if kw in (title + "\n" + content)]
    doc_id = extract_doc_id(title, content, url)
    case_type = "操纵证券市场" if keywords else "其他证券违法"

    doc = {
        "doc_id": doc_id,
        "title": title or hint,
        "url": url,
        "publish_date": pub_date,
        "content": content,
        "case_type": case_type,
        "keywords_matched": keywords,
        "crawl_time": datetime.now().isoformat(),
    }

    year = pub_date[:4] if pub_date and pub_date[:4].isdigit() else "unknown"
    year_dir = CSRC_DIR / year
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{doc_id}.json"

    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        P(f"    保存: {doc_id} [{case_type}]")
    else:
        P(f"    已存在: {doc_id}")

    return doc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(GAP_JSON, "r", encoding="utf-8") as f:
        gap = json.load(f)

    cat_b = gap["cat_b"]
    cat_c = gap["cat_c"]

    session = requests.Session()
    session.headers.update(HEADERS)

    existing_urls = set()
    for f in CSRC_DIR.rglob("*.json"):
        if f.name != "index.json":
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                existing_urls.add(d.get("url", ""))
            except Exception:
                pass

    results = {"b_results": [], "c_results": [], "stats": {}}
    crawled_count = 0

    P("=" * 80)
    P(f"B 类案例搜索+爬取 ({len(cat_b)} 个)")
    P("=" * 80)

    for i, case in enumerate(cat_b, 1):
        name = case["stock_name"]
        pattern = case.get("search_pattern", "")
        P(f"\n[B-{i}/{len(cat_b)}] {case['stock_code']} {name}  搜索: {pattern}")

        search_results = search_for_case(name, pattern)
        case_result = {**case, "search_hits": len(search_results), "crawled": False}

        if search_results:
            best = search_results[0]
            P(f"  找到: {best['title'][:60]}")
            P(f"  URL: {best['url']}")

            if best["url"] not in existing_urls:
                doc = crawl_and_save(best["url"], session, f"{name} {pattern}")
                if doc:
                    case_result["crawled"] = True
                    case_result["crawled_doc_id"] = doc["doc_id"]
                    case_result["crawled_url"] = doc["url"]
                    case_result["case_type"] = doc["case_type"]
                    crawled_count += 1
            else:
                P(f"  已存在于本地")
                case_result["crawled"] = True
                case_result["already_existed"] = True
        else:
            P(f"  未找到匹配")

        results["b_results"].append(case_result)
        time.sleep(DELAY)

    P()
    P("=" * 80)
    P(f"C 类案例搜索+爬取 ({len(cat_c)} 个)")
    P("=" * 80)

    for i, case in enumerate(cat_c, 1):
        name = case["stock_name"]
        P(f"\n[C-{i}/{len(cat_c)}] {case['stock_code']} {name}")

        search_results = search_for_case(name, None)
        case_result = {**case, "search_hits": len(search_results), "crawled": False}

        if search_results:
            best = search_results[0]
            P(f"  找到: {best['title'][:60]}")

            if best["url"] not in existing_urls:
                doc = crawl_and_save(best["url"], session, name)
                if doc:
                    case_result["crawled"] = True
                    case_result["crawled_doc_id"] = doc["doc_id"]
                    case_result["crawled_url"] = doc["url"]
                    case_result["case_type"] = doc["case_type"]
                    crawled_count += 1
            else:
                P(f"  已存在于本地")
                case_result["crawled"] = True
                case_result["already_existed"] = True
        else:
            P(f"  未找到匹配")

        results["c_results"].append(case_result)
        time.sleep(DELAY)

    b_found = sum(1 for r in results["b_results"] if r["crawled"])
    c_found = sum(1 for r in results["c_results"] if r["crawled"])

    results["stats"] = {
        "b_total": len(cat_b),
        "b_found": b_found,
        "c_total": len(cat_c),
        "c_found": c_found,
        "new_docs_crawled": crawled_count,
    }

    P()
    P("=" * 80)
    P("总结")
    P("=" * 80)
    P(f"B 类: {b_found}/{len(cat_b)} 找到并爬取")
    P(f"C 类: {c_found}/{len(cat_c)} 找到并爬取")
    P(f"新增文档: {crawled_count}")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    P(f"结果保存至: {RESULT_PATH}")

    # Update CSRC index
    update_csrc_index()


def update_csrc_index():
    all_docs = []
    for f in CSRC_DIR.rglob("*.json"):
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

    manip = sum(1 for d in all_docs if d.get("case_type") == "操纵证券市场")
    index = {
        "total": len(all_docs),
        "manipulation_cases": manip,
        "other_cases": len(all_docs) - manip,
        "update_time": datetime.now().isoformat(),
        "files": all_docs,
    }
    with open(CSRC_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    P(f"索引更新: {len(all_docs)} 文档，操纵类 {manip}")


if __name__ == "__main__":
    main()
