# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 60)

path1 = r'c:\Users\wangm\OneDrive - Audencia\毕业论文\操作处罚案例2012-2024.xlsx'
path2 = r'c:\Users\wangm\OneDrive - Audencia\毕业论文\2024年以后的操作处罚案例记录.xlsx'

def parse_mixed_dates(series):
    """Handle both datetime and Excel serial number dates."""
    result = pd.Series(pd.NaT, index=series.index)
    for i, val in series.items():
        if pd.isna(val):
            continue
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            result[i] = pd.Timestamp(val)
        elif isinstance(val, str):
            try:
                result[i] = pd.to_datetime(val)
            except:
                pass
        elif isinstance(val, (int, float)):
            try:
                result[i] = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(val))
            except:
                pass
    return result

# ================================================================
print("=" * 80)
print("文件1: 操作处罚案例2012-2024.xlsx")
print("=" * 80)
df1 = pd.read_excel(path1, sheet_name='Sheet1')
print(f"总行数: {len(df1)}")

df1['开始日期_parsed'] = parse_mixed_dates(df1['开始日期'])
df1['结束日期_parsed'] = parse_mixed_dates(df1['结束日期'])
df1['处罚日期_parsed'] = parse_mixed_dates(df1['处罚宣告日'])

has_code = df1['股票代码'].notna()
has_dates = df1['开始日期_parsed'].notna() & df1['结束日期_parsed'].notna()
valid1 = df1[has_code & has_dates].copy()
valid1['duration'] = (valid1['结束日期_parsed'] - valid1['开始日期_parsed']).dt.days

print(f"有股票代码: {has_code.sum()}")
print(f"有完整日期: {has_dates.sum()}")
print(f"同时有代码和日期: {len(valid1)}")
print(f"唯一股票代码: {valid1['股票代码'].nunique()}")

print(f"\n日期范围:")
print(f"  操纵开始: {valid1['开始日期_parsed'].min().date()} ~ {valid1['开始日期_parsed'].max().date()}")
print(f"  操纵结束: {valid1['结束日期_parsed'].min().date()} ~ {valid1['结束日期_parsed'].max().date()}")
print(f"  处罚宣告: {valid1['处罚日期_parsed'].min().date()} ~ {valid1['处罚日期_parsed'].max().date()}")

print(f"\n操纵持续天数统计:")
print(f"  均值: {valid1['duration'].mean():.1f} 天")
print(f"  中位数: {valid1['duration'].median():.1f} 天")
print(f"  最小: {valid1['duration'].min()} 天")
print(f"  最大: {valid1['duration'].max()} 天")
print(f"  分布:")
bins = [0, 10, 30, 60, 120, 365, 1000, 99999]
labels = ['<10天', '10-30天', '30-60天', '60-120天', '120-365天', '365-1000天', '>1000天']
valid1['dur_bin'] = pd.cut(valid1['duration'], bins=bins, labels=labels, right=False)
print(valid1['dur_bin'].value_counts().sort_index().to_string())

# Exchange type
print(f"\n交易所分布:")
codes = valid1['股票代码'].dropna()
for suffix in ['SH', 'SZ', 'BJ', 'NQ']:
    cnt = codes.str.endswith(f'.{suffix}').sum()
    if cnt > 0:
        print(f"  {suffix}: {cnt}")
other = codes[~codes.str.contains(r'\.(SH|SZ|BJ|NQ)$', na=False)]
if len(other) > 0:
    print(f"  其他: {len(other)}")
    print(f"    示例: {other.head(5).tolist()}")

# A-share only
ashare1 = valid1[valid1['股票代码'].str.contains(r'\.(SH|SZ)$', na=False)].copy()
print(f"\n沪深A股记录: {len(ashare1)}, 唯一股票: {ashare1['股票代码'].nunique()}")

# ================================================================
print()
print("=" * 80)
print("文件2: 2024年以后的操作处罚案例记录.xlsx (主Sheet)")
print("=" * 80)
xl2 = pd.ExcelFile(path2)
df2 = pd.read_excel(path2, sheet_name=0)
print(f"总行数: {len(df2)}")

df2['开始日期_parsed'] = parse_mixed_dates(df2['开始日期'])
df2['结束日期_parsed'] = parse_mixed_dates(df2['结束日期'])
df2['处罚日期_parsed'] = parse_mixed_dates(df2['处罚宣告日'])

has_code2 = df2['股票代码'].notna()
has_dates2 = df2['开始日期_parsed'].notna() & df2['结束日期_parsed'].notna()
valid2 = df2[has_code2 & has_dates2].copy()
valid2['duration'] = (valid2['结束日期_parsed'] - valid2['开始日期_parsed']).dt.days

print(f"有股票代码: {has_code2.sum()}")
print(f"有完整日期: {has_dates2.sum()}")
print(f"同时有代码和日期: {len(valid2)}")
print(f"唯一股票代码: {valid2['股票代码'].nunique()}")

# Non-manipulation notes filter
non_manip_keywords = ['并非操纵', '信息披露', '期货合约', '操纵基金', '从业人员违法', '保荐未尽责', '超比例持有', '公司信息造假', '未按规定披露', '未公开发行']
def is_non_manipulation(row):
    note = row.get('备注', '')
    if pd.isna(note):
        return False
    return any(kw in str(note) for kw in non_manip_keywords)

non_manip_mask = df2.apply(is_non_manipulation, axis=1)
print(f"\n备注标记为非操纵行为: {non_manip_mask.sum()} 条")
print("  (包括: 信息披露违规、期货操纵、基金操纵、超比例持有等)")

# Filter valid2 to only manipulation cases
manip2 = valid2[~valid2.index.isin(df2[non_manip_mask].index)].copy()
print(f"排除非操纵后有效记录: {len(manip2)}")

# Check for insider trading notes
insider_mask = df2['备注'].str.contains('内幕交易', na=False)
print(f"标记为内幕交易: {insider_mask.sum()} 条")

ashare2 = manip2[manip2['股票代码'].str.contains(r'\.(SH|SZ)$', na=False)].copy()
print(f"沪深A股操纵记录: {len(ashare2)}, 唯一股票: {ashare2['股票代码'].nunique()}")

# ================================================================
print()
print("=" * 80)
print("与当前实验对比 & 可用性分析")
print("=" * 80)

# Current 23 cases info
print("\n当前实验: 23个案例, 窗口长度60交易日, 步长5")
print("关键要求:")
print("  1. 必须有A股代码(SH/SZ)")
print("  2. 必须有明确的操纵开始/结束日期")
print("  3. 操纵持续时间需足够生成滑动窗口(理想>=60交易日≈84自然日)")
print("  4. 必须能获取对应时期的日线行情数据")
print("  5. 必须是市场操纵行为(影响股价)")
print()

# Use file1 as it seems cleaner, analyze usable cases
print("--- 文件1可用案例筛选 ---")
# Step 1: A-share only
step1 = ashare1.copy()
print(f"Step 1 - 沪深A股: {len(step1)} 条, {step1['股票代码'].nunique()} 只股票")

# Step 2: Duration >= 30 calendar days (about 20 trading days)
step2_30 = step1[step1['duration'] >= 30].copy()
print(f"Step 2a - 持续>=30自然日: {len(step2_30)} 条, {step2_30['股票代码'].nunique()} 只股票")

step2_60 = step1[step1['duration'] >= 60].copy()
print(f"Step 2b - 持续>=60自然日: {len(step2_60)} 条, {step2_60['股票代码'].nunique()} 只股票")

step2_90 = step1[step1['duration'] >= 90].copy()
print(f"Step 2c - 持续>=90自然日: {len(step2_90)} 条, {step2_90['股票代码'].nunique()} 只股票")

# Step 3: Show the long-duration cases
print(f"\n持续>=60自然日的案例列表(文件1):")
for _, row in step2_60.sort_values('duration', ascending=False).iterrows():
    note = f" | 备注:{row['备注']}" if pd.notna(row['备注']) else ""
    print(f"  {row['股票代码']} {row['股票名称']:6s} | {str(row['开始日期_parsed'].date()):10s} ~ {str(row['结束日期_parsed'].date()):10s} | {row['duration']:4d}天 | 处罚:{str(row['处罚日期_parsed'].date()) if pd.notna(row['处罚日期_parsed']) else 'N/A'}{note}")

# ================================================================
print()
print("--- 文件2额外案例(文件1中没有的) ---")
# Find cases in file2 but not in file1
codes1 = set(ashare1['股票代码'].unique())
extra2 = ashare2[~ashare2['股票代码'].isin(codes1)].copy()
extra2['duration'] = (extra2['结束日期_parsed'] - extra2['开始日期_parsed']).dt.days
print(f"文件2独有的A股记录: {len(extra2)}, 唯一股票: {extra2['股票代码'].nunique()}")
if len(extra2) > 0:
    extra2_long = extra2[extra2['duration'] >= 60]
    print(f"其中持续>=60天: {len(extra2_long)}")
    for _, row in extra2_long.sort_values('duration', ascending=False).iterrows():
        note = f" | 备注:{row['备注']}" if pd.notna(row['备注']) else ""
        print(f"  {row['股票代码']} {row['股票名称']:6s} | {str(row['开始日期_parsed'].date()):10s} ~ {str(row['结束日期_parsed'].date()):10s} | {row['duration']:4d}天{note}")

# ================================================================
print()
print("=" * 80)
print("总结")
print("=" * 80)
total_ashare_60 = len(step2_60)
total_ashare_90 = len(step2_90)
total_ashare_30 = len(step2_30)
print(f"""
文件1共 {len(df1)} 条记录, 文件2共 {len(df2)} 条记录
文件2是文件1的扩展版(包含更多备注标注和非操纵行为标记)

可用性评估:
  沪深A股且有完整日期的操纵记录: {len(ashare1)} 条
  其中持续>=30自然日: {total_ashare_30} 条 ({step2_30['股票代码'].nunique()} 只股票)
  其中持续>=60自然日: {total_ashare_60} 条 ({step2_60['股票代码'].nunique()} 只股票)
  其中持续>=90自然日: {total_ashare_90} 条 ({step2_90['股票代码'].nunique()} 只股票)

当前实验使用23个案例, 窗口长度60交易日(约84自然日)
如以60自然日为筛选门槛, 可从这些数据中获得 {step2_60['股票代码'].nunique()} 只不同股票的操纵案例
""")

# Quick check: duration distribution for A-share cases
print("沪深A股案例持续天数分布:")
ashare1['dur_bin'] = pd.cut(ashare1['duration'], bins=bins, labels=labels, right=False)
print(ashare1['dur_bin'].value_counts().sort_index().to_string())

print(f"\n短期操纵(<30天)案例数: {(ashare1['duration'] < 30).sum()}")
print(f"这些短期案例的中位持续天数: {ashare1[ashare1['duration'] < 30]['duration'].median()}")
print(f"这些短期案例中0天(单日操纵): {(ashare1['duration'] == 0).sum()}")
