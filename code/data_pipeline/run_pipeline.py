#!/usr/bin/env python
"""
数据基建主入口

统一管理数据采集、提取、标注和验证的完整流程。
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import log


def run_csrc_crawler_sync(args):
    """运行证监会爬虫（同步版，适用于简单页面）"""
    from crawlers.csrc_crawler import CSRCCrawler
    
    log.info("启动证监会处罚决定书爬虫（同步版）...")
    crawler = CSRCCrawler()
    saved_files = crawler.crawl(max_pages=args.max_pages)
    log.info(f"爬取完成，共保存 {len(saved_files)} 个文档")


def run_csrc_crawler_async(args):
    """运行证监会爬虫（异步Playwright版，推荐）"""
    from crawlers.csrc_crawler_async import CSRCAsyncCrawler
    
    log.info("启动证监会处罚决定书爬虫（Playwright异步版）...")
    
    async def _run():
        crawler = CSRCAsyncCrawler()
        count = await crawler.crawl(
            max_pages=args.max_pages,
            max_docs=getattr(args, 'max_docs', 200)
        )
        crawler.save_index()
        return count
    
    count = asyncio.run(_run())
    log.info(f"爬取完成，共保存 {count} 个文档")


def run_csrc_import(args):
    """手动导入证监会文书"""
    from crawlers.csrc_manual_import import CSRCManualImporter
    
    importer = CSRCManualImporter()
    
    if args.import_dir:
        log.info(f"从目录批量导入: {args.import_dir}")
        docs = importer.import_from_directory(
            args.import_dir,
            pattern=args.import_pattern or "*.txt"
        )
        log.info(f"导入完成，共 {len(docs)} 个文档")
    elif args.import_file:
        log.info(f"从文件导入: {args.import_file}")
        doc = importer.import_from_file(args.import_file)
        if doc:
            log.info(f"导入成功: {doc.doc_id}")
        else:
            log.warning("导入失败")
    else:
        # 交互式模式
        from crawlers.csrc_manual_import import main as interactive_main
        interactive_main()


def run_penalty_extractor(args):
    """运行信息提取"""
    from extractors.penalty_extractor import PenaltyExtractor
    
    log.info("启动处罚决定书信息提取...")
    extractor = PenaltyExtractor(llm_provider=args.llm_provider)
    input_dir = getattr(args, 'input_dir', None) or "data/raw/csrc_penalties"
    min_conf = getattr(args, 'min_confidence', 0.5)
    cases = extractor.batch_extract(input_dir, min_confidence=min_conf)
    log.info(f"提取完成，共 {len(cases)} 个有效案件")


def run_stock_data(args):
    """运行A股数据获取"""
    from crawlers.stock_data import AStockDataFetcher
    
    log.info("启动A股数据获取...")
    fetcher = AStockDataFetcher()
    
    cases_file = getattr(args, 'cases_file', None)
    if cases_file:
        stats = fetcher.batch_fetch_from_cases(cases_file)
        log.info(f"获取完成: {stats}")
    else:
        log.warning("请指定 --cases-file 参数")
    
    fetcher.close()


def run_crypto_data(args):
    """运行加密货币数据下载"""
    from crawlers.crypto_data import CryptoDataFetcher
    
    log.info("启动加密货币数据下载...")
    fetcher = CryptoDataFetcher()
    results = fetcher.fetch_all_symbols()
    log.info(f"下载完成，成功: {len(results)} 个交易对")


def run_crypto_labeler(args):
    """运行加密货币标注"""
    from processors.crypto_labeler import CryptoLabeler
    
    log.info("启动加密货币操纵标注...")
    labeler = CryptoLabeler()
    results = labeler.label_all_symbols()
    log.info(f"标注完成，共处理 {len(results)} 个交易对")


def run_validator(args):
    """运行数据质量验证"""
    from processors.data_validator import DataValidator
    
    log.info("启动数据质量验证...")
    validator = DataValidator()
    reports = validator.run_full_validation()
    validator.print_summary(reports)


def run_full_pipeline(args):
    """运行完整流程"""
    log.info("=" * 60)
    log.info("启动完整数据基建流程")
    log.info("=" * 60)
    
    # 1. 证监会爬虫（默认使用异步版）
    if not args.skip_csrc:
        log.info("\n【步骤1/6】爬取证监会处罚决定书（Playwright异步版）")
        try:
            run_csrc_crawler_async(args)
        except Exception as e:
            log.warning(f"异步爬虫失败: {e}，尝试同步版...")
            try:
                run_csrc_crawler_sync(args)
            except Exception as e2:
                log.error(f"同步爬虫也失败: {e2}")
                log.error("请使用手动导入: python run_pipeline.py import --import-dir <目录>")
    
    # 2. 信息提取
    if not args.skip_extract:
        log.info("\n【步骤2/6】提取结构化信息")
        args.input_dir = "data/raw/csrc_penalties"
        run_penalty_extractor(args)
    
    # 3. A股数据
    if not args.skip_stock:
        log.info("\n【步骤3/6】获取A股行情数据")
        args.cases_file = "data/processed/extracted_cases/extraction_summary.json"
        run_stock_data(args)
    
    # 4. 加密货币数据
    if not args.skip_crypto:
        log.info("\n【步骤4/6】下载加密货币数据")
        run_crypto_data(args)
    
    # 5. 加密货币标注
    if not args.skip_label:
        log.info("\n【步骤5/6】标注加密货币操纵样本")
        run_crypto_labeler(args)
    
    # 6. 数据验证
    log.info("\n【步骤6/6】数据质量验证")
    run_validator(args)
    
    log.info("\n" + "=" * 60)
    log.info("完整流程执行完毕！")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="A股操纵识别研究 - 数据基建工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行完整流程
  python run_pipeline.py full

  # 仅爬取证监会文书（Playwright异步版）
  python run_pipeline.py csrc --max-pages 50

  # 手动导入文书
  python run_pipeline.py import --import-dir /path/to/txt_files
  python run_pipeline.py import                          # 交互式

  # 仅运行信息提取
  python run_pipeline.py extract --llm-provider gemini

  # 仅下载加密货币数据
  python run_pipeline.py crypto

  # 仅运行标注
  python run_pipeline.py label

  # 数据质量检查
  python run_pipeline.py validate
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 证监会爬虫
    csrc_parser = subparsers.add_parser("csrc", help="爬取证监会处罚决定书")
    csrc_parser.add_argument("--max-pages", type=int, default=100, help="最大爬取页数")
    csrc_parser.add_argument("--max-docs", type=int, default=200, help="最大文档数")
    csrc_parser.add_argument("--sync", action="store_true", help="使用同步版爬虫")
    
    # 手动导入
    import_parser = subparsers.add_parser("import", help="手动导入证监会文书")
    import_parser.add_argument("--import-dir", help="待导入的文本文件目录")
    import_parser.add_argument("--import-file", help="单个文件路径")
    import_parser.add_argument("--import-pattern", default="*.txt", help="文件匹配模式")
    
    # 信息提取
    extract_parser = subparsers.add_parser("extract", help="提取结构化信息")
    extract_parser.add_argument("--input-dir", default="data/raw/csrc_penalties", help="输入目录")
    extract_parser.add_argument("--llm-provider", default=None, help="LLM提供商")
    extract_parser.add_argument("--min-confidence", type=float, default=0.5, help="最小置信度")
    
    # A股数据
    stock_parser = subparsers.add_parser("stock", help="获取A股行情数据")
    stock_parser.add_argument("--cases-file", help="案件汇总文件路径")
    
    # 加密货币数据
    subparsers.add_parser("crypto", help="下载加密货币数据")
    
    # 加密货币标注
    subparsers.add_parser("label", help="标注加密货币操纵样本")
    
    # 数据验证
    subparsers.add_parser("validate", help="数据质量验证")
    
    # 完整流程
    full_parser = subparsers.add_parser("full", help="运行完整流程")
    full_parser.add_argument("--max-pages", type=int, default=100, help="CSRC最大爬取页数")
    full_parser.add_argument("--max-docs", type=int, default=200, help="CSRC最大文档数")
    full_parser.add_argument("--llm-provider", default=None, help="LLM提供商")
    full_parser.add_argument("--min-confidence", type=float, default=0.5, help="最小置信度")
    full_parser.add_argument("--skip-csrc", action="store_true", help="跳过CSRC爬取")
    full_parser.add_argument("--skip-extract", action="store_true", help="跳过信息提取")
    full_parser.add_argument("--skip-stock", action="store_true", help="跳过A股数据")
    full_parser.add_argument("--skip-crypto", action="store_true", help="跳过加密货币数据")
    full_parser.add_argument("--skip-label", action="store_true", help="跳过加密货币标注")
    
    args = parser.parse_args()
    
    if args.command == "csrc":
        if args.sync:
            run_csrc_crawler_sync(args)
        else:
            run_csrc_crawler_async(args)
    elif args.command == "import":
        run_csrc_import(args)
    elif args.command == "extract":
        run_penalty_extractor(args)
    elif args.command == "stock":
        run_stock_data(args)
    elif args.command == "crypto":
        run_crypto_data(args)
    elif args.command == "label":
        run_crypto_labeler(args)
    elif args.command == "validate":
        run_validator(args)
    elif args.command == "full":
        run_full_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
