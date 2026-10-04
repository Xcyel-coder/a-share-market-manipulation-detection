"""
加密货币数据下载模块

从Binance等交易所获取历史K线数据，用于预训练数据集构建。
"""
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config


@dataclass
class CryptoDataConfig:
    """加密货币数据配置"""
    symbol: str              # 交易对
    interval: str            # K线周期
    start_date: str          # 开始日期
    end_date: str            # 结束日期


class CryptoDataFetcher:
    """加密货币数据获取器"""
    
    # Binance K线周期映射
    INTERVAL_MAP = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000
    }
    
    def __init__(self):
        """初始化"""
        self.config = config.crypto_config
        self.symbols = self.config.get("symbols", ["BTCUSDT"])
        self.interval = self.config.get("interval", "1m")
        self.start_date = self.config.get("start_date", "2021-01-01")
        self.end_date = self.config.get("end_date", "2024-12-01")
        
        # 输出目录
        self.output_dir = Path(config.paths_config.get("raw_data", "data/raw")) / "crypto"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化API客户端
        self._init_client()
        
        log.info(f"加密货币数据获取器初始化完成，交易对数: {len(self.symbols)}")
    
    def _init_client(self):
        """初始化API客户端"""
        self.ccxt_available = False
        self.binance_available = False
        
        # 尝试ccxt
        try:
            import ccxt
            self.ccxt = ccxt
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            self.ccxt_available = True
            log.info("ccxt客户端初始化成功")
        except ImportError:
            log.warning("ccxt未安装")
        except Exception as e:
            log.warning(f"ccxt初始化失败: {e}")
        
        # 尝试python-binance
        try:
            from binance.client import Client
            self.binance_client = Client()  # 公开API不需要密钥
            self.binance_available = True
            log.info("Binance客户端初始化成功")
        except ImportError:
            log.warning("python-binance未安装")
        except Exception as e:
            log.warning(f"Binance客户端初始化失败: {e}")
    
    def _date_to_timestamp(self, date_str: str) -> int:
        """日期转时间戳（毫秒）"""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)
    
    def _timestamp_to_date(self, ts: int) -> str:
        """时间戳转日期"""
        dt = datetime.fromtimestamp(ts / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def fetch_klines_ccxt(
        self,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        使用ccxt获取K线数据
        
        Args:
            symbol: 交易对
            interval: K线周期
            start_ts: 开始时间戳
            end_ts: 结束时间戳
            limit: 单次请求数量
            
        Returns:
            DataFrame或None
        """
        if not self.ccxt_available:
            return None
        
        all_data = []
        current_ts = start_ts
        interval_ms = self.INTERVAL_MAP.get(interval, 60000)
        
        pbar = tqdm(total=(end_ts - start_ts) // interval_ms, desc=f"获取{symbol}")
        
        while current_ts < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=interval,
                    since=current_ts,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                current_ts = ohlcv[-1][0] + interval_ms
                pbar.update(len(ohlcv))
                
                # 控制请求频率
                time.sleep(0.1)
                
            except Exception as e:
                log.warning(f"ccxt请求失败: {e}")
                time.sleep(1)
                continue
        
        pbar.close()
        
        if not all_data:
            return None
        
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        
        return df
    
    def fetch_klines_binance(
        self,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int
    ) -> Optional[pd.DataFrame]:
        """
        使用python-binance获取K线数据
        
        Args:
            symbol: 交易对
            interval: K线周期
            start_ts: 开始时间戳
            end_ts: 结束时间戳
            
        Returns:
            DataFrame或None
        """
        if not self.binance_available:
            return None
        
        try:
            klines = self.binance_client.get_historical_klines(
                symbol,
                interval,
                start_ts,
                end_ts
            )
            
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            
            # 只保留需要的列
            df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'symbol']]
            
            return df
            
        except Exception as e:
            log.error(f"Binance请求失败: {e}")
            return None
    
    def fetch_symbol_data(
        self,
        symbol: str,
        interval: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        获取单个交易对的完整数据
        
        Args:
            symbol: 交易对
            interval: K线周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame或None
        """
        interval = interval or self.interval
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        start_ts = self._date_to_timestamp(start_date)
        end_ts = self._date_to_timestamp(end_date)
        
        log.info(f"获取数据: {symbol}, {start_date} - {end_date}, 周期: {interval}")
        
        # 优先使用python-binance（更稳定）
        df = self.fetch_klines_binance(symbol, interval, start_ts, end_ts)
        
        # 备选ccxt
        if df is None:
            df = self.fetch_klines_ccxt(symbol, interval, start_ts, end_ts)
        
        if df is not None:
            log.info(f"获取成功: {symbol}, 共 {len(df)} 条记录")
        else:
            log.warning(f"获取失败: {symbol}")
        
        return df
    
    def save_symbol_data(self, symbol: str, df: pd.DataFrame) -> str:
        """
        保存交易对数据
        
        Args:
            symbol: 交易对
            df: 数据
            
        Returns:
            保存路径
        """
        # 按年份分割保存
        df['year'] = df['datetime'].dt.year
        
        symbol_dir = self.output_dir / symbol.replace("/", "_")
        symbol_dir.mkdir(exist_ok=True)
        
        saved_files = []
        for year, year_df in df.groupby('year'):
            file_path = symbol_dir / f"{year}.parquet"
            year_df.drop(columns=['year']).to_parquet(file_path, index=False)
            saved_files.append(str(file_path))
            log.info(f"保存: {file_path}, {len(year_df)} 条")
        
        # 保存元数据
        meta = {
            "symbol": symbol,
            "interval": self.interval,
            "start_date": df['datetime'].min().isoformat(),
            "end_date": df['datetime'].max().isoformat(),
            "total_records": len(df),
            "files": saved_files,
            "save_time": datetime.now().isoformat()
        }
        
        meta_path = symbol_dir / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        return str(symbol_dir)
    
    def fetch_all_symbols(self) -> Dict[str, str]:
        """
        获取所有配置的交易对数据
        
        Returns:
            {symbol: save_path} 字典
        """
        results = {}
        
        for symbol in self.symbols:
            try:
                df = self.fetch_symbol_data(symbol)
                if df is not None and not df.empty:
                    save_path = self.save_symbol_data(symbol, df)
                    results[symbol] = save_path
            except Exception as e:
                log.error(f"处理 {symbol} 失败: {e}")
                continue
        
        # 保存汇总
        summary = {
            "total_symbols": len(results),
            "symbols": list(results.keys()),
            "fetch_time": datetime.now().isoformat(),
            "config": {
                "interval": self.interval,
                "start_date": self.start_date,
                "end_date": self.end_date
            }
        }
        
        summary_path = self.output_dir / "fetch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        log.info(f"全部获取完成，成功: {len(results)}/{len(self.symbols)}")
        return results


def main():
    """主函数"""
    fetcher = CryptoDataFetcher()
    
    # 获取所有配置的交易对
    results = fetcher.fetch_all_symbols()
    
    print(f"\n获取完成！")
    print(f"成功获取: {len(results)} 个交易对")
    print(f"数据保存在: {fetcher.output_dir}")


if __name__ == "__main__":
    main()

