"""
加密货币操纵样本标注模块

基于统计规则自动标注"疑似操纵"样本，用于构建预训练数据集。
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log
from utils.config_loader import config


@dataclass
class ManipulationLabel:
    """操纵标签"""
    label_id: str                     # 标签ID
    symbol: str                       # 交易对
    start_time: str                   # 开始时间
    end_time: str                     # 结束时间
    manipulation_type: str            # 操纵类型
    
    # 拉高阶段特征
    pump_start_price: float           # 拉高起始价格
    pump_peak_price: float            # 最高价格
    pump_return: float                # 拉高涨幅
    pump_duration_minutes: int        # 拉高持续时间
    pump_volume_ratio: float          # 成交量放大倍数
    
    # 出货阶段特征
    dump_end_price: float             # 出货结束价格
    dump_return: float                # 回落幅度
    dump_duration_minutes: int        # 回落持续时间
    
    # 整体特征
    total_duration_minutes: int       # 总持续时间
    avg_volume: float                 # 平均成交量
    max_volume: float                 # 最大成交量
    
    # 标注元数据
    confidence: float                 # 置信度
    rule_triggered: List[str]         # 触发的规则


@dataclass
class LabelingConfig:
    """标注配置"""
    # 拉高检测
    price_surge_threshold: float = 0.10      # 涨幅阈值（10%）
    price_surge_window: int = 5              # 检测窗口（分钟）
    volume_multiplier: float = 5.0           # 成交量放大倍数
    
    # 回落检测
    price_drop_threshold: float = 0.50       # 回落阈值（50%回吐涨幅）
    price_drop_window: int = 30              # 回落窗口（分钟）
    
    # 最小持续时间
    min_pump_duration: int = 2               # 最小拉高时间（分钟）
    min_dump_duration: int = 5               # 最小回落时间（分钟）
    
    # 成交量计算
    volume_baseline_window: int = 60         # 成交量基线计算窗口（分钟）


class CryptoLabeler:
    """加密货币操纵样本标注器"""
    
    def __init__(self):
        """初始化"""
        labeling_config = config.labeling_config
        pump_dump_config = labeling_config.get("pump_dump", {})
        
        self.config = LabelingConfig(
            price_surge_threshold=pump_dump_config.get("price_surge_threshold", 0.10),
            price_surge_window=pump_dump_config.get("price_surge_window", 5),
            volume_multiplier=pump_dump_config.get("volume_multiplier", 5.0),
            price_drop_threshold=pump_dump_config.get("price_drop_threshold", 0.50),
            price_drop_window=pump_dump_config.get("price_drop_window", 30)
        )
        
        # 输入输出目录
        self.input_dir = Path(config.paths_config.get("raw_data", "data/raw")) / "crypto"
        self.output_dir = Path(config.paths_config.get("labeled_data", "data/labeled")) / "crypto"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"标注器初始化完成，涨幅阈值: {self.config.price_surge_threshold:.0%}")
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算收益率"""
        df = df.copy()
        df['return'] = df['close'].pct_change()
        df['return_5m'] = df['close'].pct_change(5)   # 5分钟收益率
        df['return_15m'] = df['close'].pct_change(15)  # 15分钟收益率
        df['return_30m'] = df['close'].pct_change(30)  # 30分钟收益率
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量特征"""
        df = df.copy()
        
        # 滚动平均成交量
        df['volume_ma'] = df['volume'].rolling(
            window=self.config.volume_baseline_window,
            min_periods=10
        ).mean()
        
        # 成交量比率
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)
        
        # 成交量Z分数
        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(
            window=self.config.volume_baseline_window
        ).std()
        
        return df
    
    def _detect_pump_phase(
        self,
        df: pd.DataFrame,
        start_idx: int
    ) -> Optional[Tuple[int, int, Dict]]:
        """
        检测拉高阶段
        
        Args:
            df: 数据
            start_idx: 起始索引
            
        Returns:
            (开始索引, 结束索引, 特征字典) 或 None
        """
        window = self.config.price_surge_window
        threshold = self.config.price_surge_threshold
        
        end_idx = min(start_idx + window, len(df) - 1)
        
        # 计算窗口内的最大涨幅
        start_price = df.iloc[start_idx]['close']
        window_data = df.iloc[start_idx:end_idx + 1]
        max_price = window_data['high'].max()
        max_price_idx = window_data['high'].idxmax()
        
        pump_return = (max_price - start_price) / start_price
        
        if pump_return >= threshold and start_price > 0:
            avg_volume_ratio = window_data['volume_ratio'].mean()
            
            if not pd.isna(avg_volume_ratio) and avg_volume_ratio >= self.config.volume_multiplier:
                features = {
                    "start_price": start_price,
                    "peak_price": max_price,
                    "pump_return": pump_return,
                    "duration": int(max_price_idx - start_idx),
                    "avg_volume_ratio": avg_volume_ratio
                }
                return start_idx, int(max_price_idx), features
        
        return None
    
    def _detect_dump_phase(
        self,
        df: pd.DataFrame,
        pump_end_idx: int,
        pump_features: Dict
    ) -> Optional[Tuple[int, Dict]]:
        """
        检测出货阶段
        
        Args:
            df: 数据
            pump_end_idx: 拉高结束索引
            pump_features: 拉高阶段特征
            
        Returns:
            (结束索引, 特征字典) 或 None
        """
        window = self.config.price_drop_window
        threshold = self.config.price_drop_threshold
        
        search_end = min(pump_end_idx + window, len(df) - 1)
        
        peak_price = pump_features["peak_price"]
        start_price = pump_features["start_price"]
        gain = peak_price - start_price
        
        # 在窗口内寻找回落点
        for idx in range(pump_end_idx + 1, search_end + 1):
            current_price = df.iloc[idx]['close']
            drop_from_peak = peak_price - current_price
            
            # 检查是否回吐了足够的涨幅
            if drop_from_peak >= gain * threshold:
                features = {
                    "end_price": current_price,
                    "dump_return": (current_price - peak_price) / peak_price,
                    "duration": idx - pump_end_idx
                }
                return idx, features
        
        return None
    
    def _create_label(
        self,
        df: pd.DataFrame,
        symbol: str,
        pump_start: int,
        pump_end: int,
        dump_end: int,
        pump_features: Dict,
        dump_features: Dict
    ) -> ManipulationLabel:
        """创建标签对象"""
        window_data = df.iloc[pump_start:dump_end + 1]
        
        label = ManipulationLabel(
            label_id=f"{symbol}_{df.iloc[pump_start]['timestamp']}",
            symbol=symbol,
            start_time=str(df.iloc[pump_start]['datetime']),
            end_time=str(df.iloc[dump_end]['datetime']),
            manipulation_type="pump_and_dump",
            
            pump_start_price=pump_features["start_price"],
            pump_peak_price=pump_features["peak_price"],
            pump_return=pump_features["pump_return"],
            pump_duration_minutes=pump_features["duration"],
            pump_volume_ratio=pump_features["avg_volume_ratio"],
            
            dump_end_price=dump_features["end_price"],
            dump_return=dump_features["dump_return"],
            dump_duration_minutes=dump_features["duration"],
            
            total_duration_minutes=dump_end - pump_start,
            avg_volume=float(window_data['volume'].mean()),
            max_volume=float(window_data['volume'].max()),
            
            confidence=self._calculate_confidence(pump_features, dump_features),
            rule_triggered=["price_surge", "volume_spike", "price_dump"]
        )
        
        return label
    
    def _calculate_confidence(
        self,
        pump_features: Dict,
        dump_features: Dict
    ) -> float:
        """计算标注置信度"""
        confidence = 0.5
        
        # 涨幅越大，置信度越高
        if pump_features["pump_return"] > 0.20:
            confidence += 0.15
        elif pump_features["pump_return"] > 0.15:
            confidence += 0.10
        
        # 成交量放大越多，置信度越高
        if pump_features["avg_volume_ratio"] > 10:
            confidence += 0.15
        elif pump_features["avg_volume_ratio"] > 7:
            confidence += 0.10
        
        # 回落越快越深，置信度越高
        if abs(dump_features["dump_return"]) > 0.15:
            confidence += 0.10
        
        if dump_features["duration"] <= 15:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def label_symbol_data(self, symbol: str) -> List[ManipulationLabel]:
        """
        标注单个交易对的数据
        
        Args:
            symbol: 交易对
            
        Returns:
            标签列表
        """
        symbol_dir = self.input_dir / symbol.replace("/", "_")
        
        if not symbol_dir.exists():
            log.warning(f"数据目录不存在: {symbol_dir}")
            return []
        
        # 读取所有年份数据
        parquet_files = list(symbol_dir.glob("*.parquet"))
        if not parquet_files:
            log.warning(f"无数据文件: {symbol_dir}")
            return []
        
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        log.info(f"加载数据: {symbol}, {len(df)} 条记录")
        
        # 计算特征
        df = self._calculate_returns(df)
        df = self._calculate_volume_features(df)
        
        # 检测操纵样本
        labels = []
        skip_until = 0
        
        for i in tqdm(range(len(df) - self.config.price_surge_window), 
                      desc=f"标注{symbol}", leave=False):
            if i < skip_until:
                continue
            
            # 检测拉高阶段
            pump_result = self._detect_pump_phase(df, i)
            if pump_result is None:
                continue
            
            pump_start, pump_end, pump_features = pump_result
            
            # 检测出货阶段
            dump_result = self._detect_dump_phase(df, pump_end, pump_features)
            if dump_result is None:
                continue
            
            dump_end, dump_features = dump_result
            
            # 创建标签
            label = self._create_label(
                df, symbol, pump_start, pump_end, dump_end,
                pump_features, dump_features
            )
            labels.append(label)
            
            # 跳过已标注区间
            skip_until = dump_end + 1
        
        log.info(f"标注完成: {symbol}, 发现 {len(labels)} 个操纵样本")
        return labels
    
    def extract_labeled_sequences(
        self,
        symbol: str,
        labels: List[ManipulationLabel],
        context_window: int = 60
    ) -> List[Dict]:
        """
        提取标注序列数据
        
        Args:
            symbol: 交易对
            labels: 标签列表
            context_window: 上下文窗口（分钟）
            
        Returns:
            序列数据列表
        """
        symbol_dir = self.input_dir / symbol.replace("/", "_")
        
        # 读取数据
        parquet_files = list(symbol_dir.glob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        sequences = []
        
        for label in labels:
            start_time = pd.to_datetime(label.start_time)
            end_time = pd.to_datetime(label.end_time)
            
            # 扩展时间窗口
            context_start = start_time - timedelta(minutes=context_window)
            context_end = end_time + timedelta(minutes=context_window)
            
            # 提取序列
            mask = (df['datetime'] >= context_start) & (df['datetime'] <= context_end)
            seq_df = df[mask].copy()
            
            if len(seq_df) < 10:
                continue
            
            # 标记操纵区间
            seq_df['is_manipulation'] = (
                (seq_df['datetime'] >= start_time) & 
                (seq_df['datetime'] <= end_time)
            ).astype(int)
            
            # 构建序列数据
            sequence = {
                "label_id": label.label_id,
                "symbol": symbol,
                "manipulation_type": label.manipulation_type,
                "confidence": label.confidence,
                "sequence_length": len(seq_df),
                "manipulation_ratio": seq_df['is_manipulation'].mean(),
                "features": {
                    "open": seq_df['open'].tolist(),
                    "high": seq_df['high'].tolist(),
                    "low": seq_df['low'].tolist(),
                    "close": seq_df['close'].tolist(),
                    "volume": seq_df['volume'].tolist(),
                    "timestamp": seq_df['timestamp'].tolist(),
                    "is_manipulation": seq_df['is_manipulation'].tolist()
                },
                "label_info": asdict(label)
            }
            
            sequences.append(sequence)
        
        return sequences
    
    def save_labels(
        self,
        symbol: str,
        labels: List[ManipulationLabel],
        sequences: List[Dict]
    ) -> Tuple[str, str]:
        """
        保存标签和序列
        
        Args:
            symbol: 交易对
            labels: 标签列表
            sequences: 序列数据
            
        Returns:
            (标签文件路径, 序列文件路径)
        """
        symbol_dir = self.output_dir / symbol.replace("/", "_")
        symbol_dir.mkdir(exist_ok=True)
        
        # 保存标签
        labels_data = [asdict(l) for l in labels]
        labels_path = symbol_dir / "labels.json"
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(labels_data, f, ensure_ascii=False, indent=2)
        
        # 保存序列（分批保存，避免单文件过大）
        batch_size = 1000
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            seq_path = symbol_dir / f"sequences_{i // batch_size}.json"
            with open(seq_path, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False)
        
        # 保存元数据
        meta = {
            "symbol": symbol,
            "total_labels": len(labels),
            "total_sequences": len(sequences),
            "config": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
            "label_time": datetime.now().isoformat()
        }
        meta_path = symbol_dir / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        log.info(f"保存完成: {symbol_dir}")
        return str(labels_path), str(symbol_dir)
    
    def label_all_symbols(self) -> Dict[str, Dict]:
        """
        标注所有交易对
        
        Returns:
            统计信息
        """
        # 获取所有已下载的交易对
        symbol_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        log.info(f"开始标注，共 {len(symbol_dirs)} 个交易对")
        
        results = {}
        total_labels = 0
        
        for symbol_dir in tqdm(symbol_dirs, desc="标注进度"):
            symbol = symbol_dir.name
            
            try:
                # 标注
                labels = self.label_symbol_data(symbol)
                
                if labels:
                    # 提取序列
                    sequences = self.extract_labeled_sequences(symbol, labels)
                    
                    # 保存
                    self.save_labels(symbol, labels, sequences)
                    
                    results[symbol] = {
                        "labels_count": len(labels),
                        "sequences_count": len(sequences)
                    }
                    total_labels += len(labels)
                    
            except Exception as e:
                log.error(f"标注失败: {symbol}, 错误: {e}")
                continue
        
        # 保存汇总
        summary = {
            "total_symbols": len(results),
            "total_labels": total_labels,
            "symbols": results,
            "config": {
                "price_surge_threshold": self.config.price_surge_threshold,
                "volume_multiplier": self.config.volume_multiplier,
                "price_drop_threshold": self.config.price_drop_threshold
            },
            "label_time": datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "labeling_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        log.info(f"全部标注完成，共 {total_labels} 个操纵样本")
        return results


def main():
    """主函数"""
    labeler = CryptoLabeler()
    
    # 标注所有交易对
    results = labeler.label_all_symbols()
    
    print(f"\n标注完成！")
    print(f"处理交易对: {len(results)} 个")
    print(f"标注结果保存在: {labeler.output_dir}")


if __name__ == "__main__":
    main()

