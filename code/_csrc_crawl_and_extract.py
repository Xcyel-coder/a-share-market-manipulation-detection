# -*- coding: utf-8 -*-
"""
CSRC 定向爬取 + LLM 提取 (合并阶段二/三/四)

从 WebSearch 发现的 CSRC URL 中爬取详情页，然后运行 LLM 提取。
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

P = lambda *a, **kw: print(*a, **kw, flush=True)

BASE = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文")
CSRC_DIR = BASE / "code" / "data_pipeline" / "data" / "raw" / "csrc_penalties"
EXTRACTED_DIR = BASE / "code" / "data_pipeline" / "data" / "processed" / "extracted_cases"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

MANIPULATION_KEYWORDS = [
    "操纵", "对倒", "对敲", "拉抬", "打压",
    "虚假申报", "撤单", "洗售", "相对委托",
    "尾市", "连续交易", "约定交易", "连续买卖",
    "集中资金优势", "持股优势", "账户组",
]

# Discovered CSRC URLs from web search.
# Format: (url, csv_id, stock_code, stock_name, csv_penalty_id)
DISCOVERED_URLS = [
    # B-class: matched via penalty_id / stock name
    ("http://www.csrc.gov.cn/csrc/c101928/c6222796/content.shtml",
     93, "600807.SH", "ST天业", "202249"),
    ("http://www.csrc.gov.cn/csrc/c101928/c1416745/content.shtml",
     42, "002647.SZ", "仁东控股", "202178"),
    ("http://www.csrc.gov.cn/csrc/c101928/c7088656/content.shtml",
     47, "002752.SZ", "昇兴股份", "202261"),
    ("http://www.csrc.gov.cn/csrc/c101928/c7412277/content.shtml",
     50, "002776.SZ", "柏堡龙", "202337"),
    ("http://www.csrc.gov.cn/csrc/c101928/c7422091/content.shtml",
     49, "002760.SZ", "凤形股份", "202355"),
    ("http://www.csrc.gov.cn/csrc/c101928/c7491797/content.shtml",
     91, "600735.SH", "新华锦", "201796"),
    ("http://www.csrc.gov.cn/csrc/c101928/c1042946/content.shtml",
     4, "000511.SZ", "银基发展", "201603"),
    # Additional manipulation cases found (may match C-class)
    ("http://www.csrc.gov.cn/shanghai/c103864/c7454310/content.shtml",
     8, "000673.SZ", "当代东方", "202354"),
    ("http://www.csrc.gov.cn/csrc/c101928/c7500836/content.shtml",
     0, None, "冯越峰操纵", None),
    ("http://www.csrc.gov.cn/csrc/c101928/c1042625/content.shtml",
     0, None, "冯志浩操纵", None),
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
                P(f"  请求失败: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def extract_content(html: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    el = soup.select_one("meta[name='ArticleTitle']")
    if el:
        title = el.get("content", "")
    if not title:
        h = soup.select_one("h2") or soup.select_one("h1")
        if h:
            title = h.get_text(strip=True)

    content = ""
    for sel in ["#ContentRegion", "div.TRS_Editor", "div.detail_content",
                "div.content_detail", "div.article-content", "article",
                "div.xxgk-detail-content", "div.content"]:
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
            content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    pub_date = ""
    el = soup.select_one("meta[name='PubDate']")
    if el:
        pub_date = el.get("content", "")
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


def crawl_and_save(url: str, session: requests.Session, hint: str) -> dict | None:
    html = fetch_page(url, session)
    if not html:
        return None
    content, title, pub_date = extract_content(html)
    if not content or len(content) < 300:
        P(f"  内容过短，跳过")
        return None

    keywords = [kw for kw in MANIPULATION_KEYWORDS if kw in (title + "\n" + content)]
    doc_id = extract_doc_id(title, content, url)
    case_type = "操纵证券市场" if keywords else "其他证券违法"

    doc = {
        "doc_id": doc_id, "title": title or hint, "url": url,
        "publish_date": pub_date, "content": content,
        "case_type": case_type, "keywords_matched": keywords,
        "crawl_time": datetime.now().isoformat(),
    }

    year = pub_date[:4] if pub_date and pub_date[:4].isdigit() else "unknown"
    year_dir = CSRC_DIR / year
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{doc_id}.json"

    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        P(f"  新保存: {doc_id} [{case_type}] {title[:50]}")
    else:
        P(f"  已存在: {doc_id}")
    return doc


# ─── LLM Extraction ──────────────────────────────────────────────────────────

def init_gemini():
    """Initialize Gemini client."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        env_path = BASE / "code" / "data_pipeline" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GOOGLE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        P("警告: 未找到 GOOGLE_API_KEY，跳过 LLM 提取")
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
            return ("old", genai_old.GenerativeModel("gemini-2.0-flash"))
        except ImportError:
            P("警告: 未安装 google-genai，跳过 LLM 提取")
            return None


EXTRACTION_PROMPT = """你是一位专业的金融法律文书分析专家。请从以下证监会行政处罚决定书中提取结构化信息。

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

请直接输出JSON，不要有其他内容。"""


def call_llm(client, content: str) -> dict | None:
    """Call Gemini to extract structured information."""
    cleaned = re.sub(r'\s+', ' ', content)
    if len(cleaned) > 8000:
        head = cleaned[:5000]
        tail = cleaned[-2500:]
        cleaned = head + "\n...[中间部分省略]...\n" + tail

    prompt = EXTRACTION_PROMPT.format(content=cleaned)

    try:
        if isinstance(client, tuple) and client[0] == "old":
            response = client[1].generate_content(
                prompt, generation_config={"temperature": 0.1, "max_output_tokens": 4096})
            text = response.text
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 4096})
            text = response.text

        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        P(f"  LLM 调用失败: {e}")
    return None


def extract_case(client, doc_path: Path) -> dict | None:
    data = json.loads(doc_path.read_text(encoding="utf-8"))
    content = data.get("content", "")
    if not content or len(content) < 300:
        return None

    extracted = call_llm(client, content)
    if not extracted:
        return None

    key_fields = ["stock_code", "stock_name", "manipulation_start_date",
                  "manipulation_end_date", "manipulation_methods",
                  "manipulation_description", "defendant_name"]
    confidence = sum(1 for f in key_fields if extracted.get(f)) / len(key_fields)

    start = extracted.get("manipulation_start_date", "")
    end = extracted.get("manipulation_end_date", "")
    duration = 0
    try:
        if start and end:
            d1 = datetime.strptime(start, "%Y-%m-%d")
            d2 = datetime.strptime(end, "%Y-%m-%d")
            duration = (d2 - d1).days + 1
    except Exception:
        pass

    case = {
        "doc_id": data.get("doc_id", doc_path.stem),
        "case_number": extracted.get("case_number", ""),
        "defendant_name": extracted.get("defendant_name", ""),
        "defendant_type": extracted.get("defendant_type", ""),
        "stock_code": str(extracted.get("stock_code", "")),
        "stock_name": extracted.get("stock_name", ""),
        "manipulation_start_date": start,
        "manipulation_end_date": end,
        "manipulation_duration_days": duration,
        "manipulation_methods": extracted.get("manipulation_methods", []),
        "manipulation_description": extracted.get("manipulation_description", ""),
        "account_count": int(extracted.get("account_count") or 0),
        "account_group_description": extracted.get("account_group_description", ""),
        "involved_amount": float(extracted.get("involved_amount") or 0),
        "illegal_profit": float(extracted.get("illegal_profit") or 0),
        "penalty_type": extracted.get("penalty_type", ""),
        "fine_amount": float(extracted.get("fine_amount") or 0),
        "confiscation_amount": float(extracted.get("confiscation_amount") or 0),
        "market_ban_years": int(extracted.get("market_ban_years") or 0),
        "source_url": data.get("url", ""),
        "publish_date": data.get("publish_date", ""),
        "extract_time": datetime.now().isoformat(),
        "confidence_score": confidence,
        "key_paragraphs": extracted.get("key_paragraphs", []),
    }
    return case


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    # ── Step 1: Crawl discovered URLs ─────────────────────────────────────
    P("=" * 80)
    P("步骤一：爬取发现的 CSRC URL")
    P("=" * 80)

    crawled_docs = []
    for url, csv_id, scode, sname, pid in DISCOVERED_URLS:
        hint = f"{scode or ''} {sname or ''}"
        P(f"\n爬取: {hint.strip()}")
        doc = crawl_and_save(url, session, hint.strip())
        if doc:
            doc["_csv_id"] = csv_id
            doc["_stock_code"] = scode
            doc["_penalty_id"] = pid
            crawled_docs.append(doc)
        time.sleep(2)

    P(f"\n爬取完成: {len(crawled_docs)} 个文档")

    # ── Step 2: LLM extraction on ALL unextracted manipulation docs ───────
    P()
    P("=" * 80)
    P("步骤二：LLM 结构化提取")
    P("=" * 80)

    client = init_gemini()
    if not client:
        P("跳过 LLM 提取 (无 API Key)")
        return

    already_extracted = set()
    for f in EXTRACTED_DIR.glob("*_extracted.json"):
        already_extracted.add(f.stem.replace("_extracted", ""))

    to_extract = []
    for f in CSRC_DIR.rglob("*.json"):
        if f.name == "index.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            doc_id = data.get("doc_id", "")
            if doc_id not in already_extracted:
                keywords = data.get("keywords_matched", [])
                case_type = data.get("case_type", "")
                if keywords or case_type == "操纵证券市场":
                    to_extract.append(f)
        except Exception:
            pass

    P(f"已提取: {len(already_extracted)}, 待提取: {len(to_extract)}")

    new_cases = []
    for i, doc_path in enumerate(to_extract, 1):
        doc_id = doc_path.stem
        P(f"\n[{i}/{len(to_extract)}] 提取: {doc_id}")

        case = extract_case(client, doc_path)
        if case:
            out_path = EXTRACTED_DIR / f"{case['doc_id']}_extracted.json"
            EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(case, fp, ensure_ascii=False, indent=2)
            P(f"  -> {case['stock_name']} {case['manipulation_start_date']}~{case['manipulation_end_date']} "
              f"置信度={case['confidence_score']:.2f}")
            new_cases.append(case)
        else:
            P(f"  -> 提取失败")

        time.sleep(1)

    P(f"\n新提取: {len(new_cases)} 个案例")

    # Update index
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
        "total": len(all_docs), "manipulation_cases": manip,
        "other_cases": len(all_docs) - manip,
        "update_time": datetime.now().isoformat(), "files": all_docs,
    }
    with open(CSRC_DIR / "index.json", "w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False, indent=2)
    P(f"索引更新: {len(all_docs)} 文档，操纵类 {manip}")


if __name__ == "__main__":
    main()
