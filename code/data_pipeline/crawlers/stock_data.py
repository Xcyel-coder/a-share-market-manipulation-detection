"""
A股行情数据获取模块

获取涉案股票的分钟级量价数据、龙虎榜数据等。
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config


@dataclass
class StockDataRequest:
    """股票数据请求"""
    stock_code: str           # 股票代码
    start_date: str           # 开始日期
    end_date: str             # 结束日期
    case_id: str              # 关联案件ID
    extend_days: int = 30     # 前后延伸天数


class AStockDataFetcher:
    """A股数据获取器"""
    
    def __init__(self):
        """初始化"""
        self.config = config.astock_config
        self.frequency = self.config.get("frequency", "1min")
        self.fields = self.config.get("fields", ["open", "high", "low", "close", "volume"])
        
        # 输出目录
        self.output_dir = Path(config.paths_config.get("raw_data", "data/raw")) / "astock"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据源
        self._init_data_source()
        
        log.info(f"A股数据获取器初始化完成，频率: {self.frequency}")
    
    def _init_data_source(self):
        """初始化数据源"""
        self.akshare_available = False
        self.baostock_available = False
        
        # 尝试导入akshare
        try:
            import akshare as ak
            self.ak = ak
            self.akshare_available = True
            log.info("akshare数据源可用")
        except ImportError:
            log.warning("akshare未安装，部分功能不可用")
        
        # 尝试导入baostock
        try:
            import baostock as bs
            self.bs = bs
            # 登录baostock
            lg = bs.login()
            if lg.error_code == '0':
                self.baostock_available = True
                log.info("baostock数据源可用")
            else:
                log.warning(f"baostock登录失败: {lg.error_msg}")
        except ImportError:
            log.warning("baostock未安装，部分功能不可用")
    
    def _normalize_stock_code(self, code: str) -> Tuple[str, str]:
        """
        标准化股票代码
        
        Args:
            code: 原始代码
            
        Returns:
            (标准代码, 市场前缀)
        """
        code = str(code).strip()
        
        # 移除已有前缀
        code = code.replace("sh.", "").replace("sz.", "")
        code = code.replace("SH", "").replace("SZ", "")
        code = code.replace(".SH", "").replace(".SZ", "")
        
        # 补齐6位
        code = code.zfill(6)
        
        # 判断市场
        if code.startswith(('6', '9')):
            return code, "sh"
        else:
            return code, "sz"
    
    def get_minute_data_akshare(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        period: str = "1"
    ) -> Optional[pd.DataFrame]:
        """
        使用akshare获取分钟数据
        
        注意：akshare 分钟数据仅支持最近5-9个交易日，对历史案例无效。
        对于历史案例，应使用日线数据作为主要数据源。
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 分钟周期（"1","5","15","30","60"）
            
        Returns:
            DataFrame或None
        """
        if not self.akshare_available:
            return None
        
        code, market = self._normalize_stock_code(stock_code)
        
        try:
            # stock_zh_a_hist_min_em 返回列：时间/开盘/收盘/最高/最低/成交量/成交额/均价
            df = self.ak.stock_zh_a_hist_min_em(
                symbol=code,
                period=period,
                start_date=f"{start_date} 09:30:00",
                end_date=f"{end_date} 15:00:00",
                adjust="qfq"
            )
            
            if df is None or df.empty:
                log.warning(f"akshare分钟数据为空（可能超出可用范围）: {code}")
                return None
            
            df = df.rename(columns={
                "时间": "datetime", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "均价": "avg_price"
            })
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            df['stock_code'] = code
            df['market'] = market
            
            log.info(f"akshare分钟数据获取成功: {code}, {len(df)} 条")
            return df
            
        except Exception as e:
            log.warning(f"akshare分钟数据不可用: {code}, 错误: {e}")
            return None
    
    def get_daily_data_baostock(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        使用baostock获取日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame或None
        """
        if not self.baostock_available:
            return None
        
        code, market = self._normalize_stock_code(stock_code)
        symbol = f"{market}.{code}"
        
        try:
            rs = self.bs.query_history_k_data_plus(
                symbol,
                "date,code,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )
            
            data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                log.warning(f"baostock返回空数据: {symbol}")
                return None
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['date'] = pd.to_datetime(df['date'])
            df['stock_code'] = code
            df['market'] = market
            
            return df
            
        except Exception as e:
            log.error(f"baostock获取数据失败: {symbol}, 错误: {e}")
            return None
    
    def get_daily_data_akshare(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        使用akshare获取日线数据（作为baostock的备选）
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（YYYYMMDD 或 YYYY-MM-DD）
            end_date: 结束日期
            
        Returns:
            DataFrame或None
        """
        if not self.akshare_available:
            return None
        
        code, market = self._normalize_stock_code(stock_code)
        
        try:
            start_fmt = start_date.replace("-", "")
            end_fmt = end_date.replace("-", "")
            
            df = self.ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
                adjust="qfq"
            )
            
            if df is None or df.empty:
                log.warning(f"akshare日线数据为空: {code}")
                return None
            
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "pctChg", "涨跌额": "change", "换手率": "turn"
            })
            df['date'] = pd.to_datetime(df['date'])
            df['stock_code'] = code
            df['market'] = market
            
            log.info(f"akshare日线数据获取成功: {code}, {len(df)} 条")
            return df
            
        except Exception as e:
            log.warning(f"akshare日线数据失败: {code}, 错误: {e}")
            return None

    def get_dragon_tiger_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        获取龙虎榜数据
        
        使用 stock_lhb_detail_daily_sina 获取指定日期范围的龙虎榜数据。
        由于该接口按日期查询，会逐日尝试获取操纵期间的上榜记录。
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame或None
        """
        if not self.akshare_available:
            return None
        
        code, _ = self._normalize_stock_code(stock_code)
        
        try:
            # stock_lhb_ggtj_sina: 龙虎榜个股统计，参数为 recent_day="5"/"10"/"30"/"60"
            df = self.ak.stock_lhb_ggtj_sina(recent_day="60")
            
            if df is None or df.empty:
                return None
            
            # 筛选目标股票
            code_col = None
            for col in df.columns:
                if "代码" in col:
                    code_col = col
                    break
            
            if code_col:
                df[code_col] = df[code_col].astype(str).str.zfill(6)
                df = df[df[code_col] == code]
            
            if df.empty:
                log.info(f"龙虎榜无此股票记录: {code}")
                return None
            
            df['stock_code'] = code
            log.info(f"获取龙虎榜数据: {code}, {len(df)} 条")
            return df
            
        except Exception as e:
            log.warning(f"获取龙虎榜数据失败: {code}, 错误: {e}")
            return None
    
    def fetch_case_data(self, request: StockDataRequest) -> Dict[str, pd.DataFrame]:
        """
        获取案件相关的全部数据（多数据源 fallback）
        
        数据优先级:
        - 日线: baostock → akshare
        - 分钟: akshare（仅近期数据可用，历史案例不可用）
        - 龙虎榜: akshare
        
        Args:
            request: 数据请求
            
        Returns:
            数据字典
        """
        log.info(f"获取案件数据: {request.case_id}, 股票: {request.stock_code}")
        
        start = datetime.strptime(request.start_date, "%Y-%m-%d") - timedelta(days=request.extend_days)
        end = datetime.strptime(request.end_date, "%Y-%m-%d") + timedelta(days=request.extend_days)
        
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        result = {}
        
        # 日线数据：baostock 优先，akshare 兜底
        daily_df = self.get_daily_data_baostock(request.stock_code, start_str, end_str)
        if daily_df is None or daily_df.empty:
            log.info("baostock日线不可用，尝试akshare...")
            daily_df = self.get_daily_data_akshare(request.stock_code, start_str, end_str)
        if daily_df is not None and not daily_df.empty:
            result['daily'] = daily_df
            log.info(f"日线数据: {len(daily_df)} 条")
        else:
            log.warning(f"日线数据获取失败: {request.stock_code}")
        
        # 分钟数据：仅尝试（历史案例大概率不可用）
        minute_df = self.get_minute_data_akshare(request.stock_code, start_str, end_str)
        if minute_df is not None and not minute_df.empty:
            result['minute'] = minute_df
            log.info(f"分钟数据: {len(minute_df)} 条")
        else:
            log.info(f"分钟数据不可用（历史案例属正常现象）: {request.stock_code}")
        
        # 龙虎榜数据
        lhb_df = self.get_dragon_tiger_data(
            request.stock_code, request.start_date, request.end_date
        )
        if lhb_df is not None and not lhb_df.empty:
            result['dragon_tiger'] = lhb_df
            log.info(f"龙虎榜数据: {len(lhb_df)} 条")
        
        return result
    
    def save_case_data(
        self,
        case_id: str,
        data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """
        保存案件数据
        
        Args:
            case_id: 案件ID
            data: 数据字典
            
        Returns:
            保存的文件路径列表
        """
        case_dir = self.output_dir / case_id
        case_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        for data_type, df in data.items():
            if df is not None and not df.empty:
                file_path = case_dir / f"{data_type}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(str(file_path))
                log.info(f"保存数据: {file_path}")
        
        # 保存元数据
        meta = {
            "case_id": case_id,
            "save_time": datetime.now().isoformat(),
            "data_types": list(data.keys()),
            "files": saved_files
        }
        meta_path = case_dir / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        return saved_files
    
    def batch_fetch_from_cases(self, cases_file: str) -> Dict[str, int]:
        """
        根据提取的案件批量获取数据
        
        Args:
            cases_file: 案件汇总文件路径
            
        Returns:
            统计信息
        """
        with open(cases_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        cases = summary.get("cases", [])
        log.info(f"开始批量获取，共 {len(cases)} 个案件")
        
        stats = {"success": 0, "failed": 0, "no_data": 0}
        
        for case in tqdm(cases, desc="获取进度"):
            stock_code = case.get("stock_code")
            if not stock_code or stock_code == "null":
                stats["no_data"] += 1
                continue
            
            request = StockDataRequest(
                stock_code=stock_code,
                start_date=case.get("manipulation_start_date", "2020-01-01"),
                end_date=case.get("manipulation_end_date", "2020-12-31"),
                case_id=case.get("doc_id", "unknown")
            )
            
            try:
                data = self.fetch_case_data(request)
                if data:
                    self.save_case_data(request.case_id, data)
                    stats["success"] += 1
                else:
                    stats["no_data"] += 1
            except Exception as e:
                log.error(f"获取失败: {request.case_id}, 错误: {e}")
                stats["failed"] += 1
        
        log.info(f"批量获取完成: {stats}")
        return stats
    
    def close(self):
        """关闭连接"""
        if self.baostock_available:
            try:
                self.bs.logout()
            except:
                pass


def main():
    """主函数"""
    fetcher = AStockDataFetcher()
    
    # 示例：获取单个案件数据
    request = StockDataRequest(
        stock_code="000001",
        start_date="2023-01-01",
        end_date="2023-06-30",
        case_id="test_case"
    )
    
    data = fetcher.fetch_case_data(request)
    if data:
        fetcher.save_case_data(request.case_id, data)
        print(f"\n获取完成！数据保存在: {fetcher.output_dir / request.case_id}")
    
    fetcher.close()


if __name__ == "__main__":
    main()

