# 数据基建模块

基于混合专家模型与跨市场辅助迁移的A股操纵识别智能体研究 - 数据采集与处理模块

## 目录结构

```
data_pipeline/
├── configs/
│   └── config.yaml             # 主配置（交易对、标注阈值等）
├── crawlers/
│   ├── csrc_crawler.py         # 证监会爬虫（同步版，简单页面）
│   ├── csrc_crawler_async.py   # 证监会爬虫（Playwright异步版，推荐）
│   ├── csrc_manual_import.py   # 证监会文书手动导入工具
│   ├── stock_data.py           # A股行情数据（akshare + baostock）
│   └── crypto_data.py          # 加密货币数据（Binance API）
├── extractors/
│   └── penalty_extractor.py    # 处罚书 → 结构化案件（LLM提取）
├── processors/
│   ├── crypto_labeler.py       # 加密货币操纵样本自动标注
│   └── data_validator.py       # 数据质量检查
├── utils/
│   ├── logger.py               # 日志（loguru）
│   └── config_loader.py        # YAML配置加载
├── data/                        # 数据目录（git忽略）
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后数据
│   └── labeled/                # 标注数据
├── .env.example                # 环境变量模板
├── requirements.txt            # Python依赖
├── run_pipeline.py             # 统一入口
└── README.md
```

## 快速开始

### 1. 环境配置

```bash
cd code/data_pipeline
python -m venv venv
source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
playwright install chromium    # Playwright浏览器（证监会爬虫需要）
```

### 2. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入 GOOGLE_API_KEY（用于处罚书信息提取）
```

### 3. 运行

```bash
# === 完整流程（推荐首次运行） ===
python run_pipeline.py full

# === 分步运行 ===

# 第1步：采集证监会处罚文书
python run_pipeline.py csrc --max-pages 50       # 自动爬取（Playwright）
python run_pipeline.py import --import-dir ./docs # 手动导入（如爬虫不可用）
python run_pipeline.py import                     # 交互式手动导入

# 第2步：LLM提取结构化信息
python run_pipeline.py extract --llm-provider gemini

# 第3步：获取A股行情数据
python run_pipeline.py stock --cases-file data/processed/extracted_cases/extraction_summary.json

# 第4步：下载加密货币数据（约1-2小时）
python run_pipeline.py crypto

# 第5步：自动标注操纵样本
python run_pipeline.py label

# 第6步：数据质量检查
python run_pipeline.py validate
```

## 数据采集策略

### A. 证监会处罚决定书（目标：≥200份操纵类）

提供**三种采集方式**，按优先级排列：

| 方式 | 命令 | 适用场景 |
|------|------|---------|
| Playwright异步爬虫 | `python run_pipeline.py csrc` | 首选，自动处理JS渲染 |
| 手动导入 | `python run_pipeline.py import` | 爬虫受限时，从浏览器复制文书文本导入 |
| 同步爬虫 | `python run_pipeline.py csrc --sync` | 备选，仅适用于静态页面 |

**手动导入流程**：
1. 在浏览器打开 [证监会行政处罚](http://www.csrc.gov.cn/csrc/c101928/zfxxgk_zdgk.shtml)
2. 找到操纵类处罚决定书，复制全文
3. 保存为 `.txt` 文件到某目录
4. 运行 `python run_pipeline.py import --import-dir <目录>`

### B. A股行情数据

- **日线数据**：baostock（首选）→ akshare（备选）
- **分钟数据**：akshare（仅最近1-2周可用，历史案例不可用）
- **龙虎榜**：akshare（`stock_lhb_ggtj_sina`）

### C. 加密货币数据（已完成采集）

| 交易对 | 记录数 | 时间范围 |
|--------|--------|---------|
| BTC/USDT | ~205万 | 2021-2024 |
| ETH/USDT | ~205万 | 2021-2024 |
| DOGE/USDT | ~205万 | 2021-2024 |
| SHIB/USDT | ~187万 | 2021-2024 |
| PEPE/USDT | ~83万 | 2023-2024 |
| FLOKI/USDT | ~83万 | 2023-2024 |
| **合计** | **~968万** | |

### D. 操纵标注规则

拉高出货（Pump & Dump）检测规则：
- 5分钟内涨幅 ≥ 10%
- 成交量放大 ≥ 5倍（相对60分钟滚动均值）
- 30分钟内回吐涨幅 ≥ 50%

规则阈值可在 `configs/config.yaml` 中调整。

## 数据目标

| 数据类型 | 目标数量 | 当前状态 |
|----------|----------|---------|
| 证监会处罚书 | ≥200份（操纵类） | 5份测试数据 |
| 提取的结构化案件 | ≥150条（置信度≥0.5） | 3条 |
| A股日线数据 | 覆盖≥80%案件 | 3个案件 |
| 加密货币K线 | ≥900万条 | ~968万条 ✓ |
| 操纵标注样本 | ≥10万条 | 待运行 |

## 常见问题

### 证监会网站打不开/爬取失败？
使用手动导入：`python run_pipeline.py import`

### akshare分钟数据获取不到？
这是正常现象——akshare仅提供最近5-9个交易日的分钟数据。历史案例请使用日线数据。

### 加密货币数据下载超时？
确保网络可访问 Binance API。可在 `config.yaml` 中调整交易对列表。

### LLM提取效果不好？
- 检查 API Key 是否配置正确
- 在 `penalty_extractor.py` 中调整 `EXTRACTION_PROMPT_TEMPLATE`
- 尝试不同的模型（gemini-2.5-flash / gpt-4o）
