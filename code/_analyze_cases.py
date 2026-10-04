# -*- coding: utf-8 -*-
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 60)

path1 = r'c:\Users\wangm\OneDrive - Audencia\毕业论文\操作处罚案例2012-2024.xlsx'
path2 = r'c:\Users\wangm\OneDrive - Audencia\毕业论文\2024年以后的操作处罚案例记录.xlsx'

print("=" * 80)
print("文件1: 操作处罚案例2012-2024.xlsx")
print("=" * 80)
df1 = pd.read_excel(path1, sheet_name='Sheet1')
print(f"总行数: {len(df1)}, 列: {list(df1.columns)}")
print()

# Parse dates properly
for col in ['开始日期', '结束日期', '处罚宣告日']:
    df1[col] = pd.to_datetime(df1[col], errors='coerce')

print("前10行:")
print(df1.head(10).to_string())
print()

# Stats
has_code = df1['股票代码'].notna()
has_start = df1['开始日期'].notna()
has_end = df1['结束日期'].notna()
print(f"有股票代码: {has_code.sum()}/{len(df1)}")
print(f"有开始日期: {has_start.sum()}/{len(df1)}")
print(f"有结束日期: {has_end.sum()}/{len(df1)}")
print(f"唯一股票代码: {df1['股票代码'].nunique()}")

# Date ranges
valid = df1[has_start & has_end].copy()
print(f"\n有完整日期范围的记录: {len(valid)}")
print(f"开始日期: {valid['开始日期'].min()} ~ {valid['开始日期'].max()}")
print(f"结束日期: {valid['结束日期'].min()} ~ {valid['结束日期'].max()}")
print(f"处罚日期: {valid['处罚宣告日'].min()} ~ {valid['处罚宣告日'].max()}")

# Duration stats
valid['duration_days'] = (valid['结束日期'] - valid['开始日期']).dt.days
print(f"\n操纵持续天数统计:")
print(f"  均值: {valid['duration_days'].mean():.1f} 天")
print(f"  中位数: {valid['duration_days'].median():.1f} 天")
print(f"  最小: {valid['duration_days'].min()} 天")
print(f"  最大: {valid['duration_days'].max()} 天")
print(f"  <10天: {(valid['duration_days'] < 10).sum()}")
print(f"  10-60天: {((valid['duration_days'] >= 10) & (valid['duration_days'] <= 60)).sum()}")
print(f"  60-365天: {((valid['duration_days'] > 60) & (valid['duration_days'] <= 365)).sum()}")
print(f"  >365天: {(valid['duration_days'] > 365).sum()}")

# Check exchange types
print(f"\n股票代码后缀分布:")
codes = df1['股票代码'].dropna()
suffixes = codes.str.extract(r'\.(\w+)')[0].value_counts()
print(suffixes.to_string())

# Check for notes
notes = df1[df1['备注'].notna()]
print(f"\n有备注的记录: {len(notes)}")
for i, row in notes.head(15).iterrows():
    print(f"  [{i}] {row['股票名称']}: {row['备注']}")

print()
print("=" * 80)
print("文件2: 2024年以后的操作处罚案例记录.xlsx")
print("=" * 80)
xl2 = pd.ExcelFile(path2)
print(f"Sheet列表: {xl2.sheet_names}")
print()

df2 = pd.read_excel(path2, sheet_name=xl2.sheet_names[0])
print(f"主Sheet '{xl2.sheet_names[0]}' 行数: {len(df2)}, 列: {list(df2.columns)}")
print()

for col in ['开始日期', '结束日期', '处罚宣告日']:
    df2[col] = pd.to_datetime(df2[col], errors='coerce')

print("前15行:")
print(df2.head(15).to_string())
print()

has_code2 = df2['股票代码'].notna()
has_start2 = df2['开始日期'].notna()
has_end2 = df2['结束日期'].notna()
complete2 = df2[has_code2 & has_start2 & has_end2].copy()
print(f"有股票代码: {has_code2.sum()}/{len(df2)}")
print(f"有完整日期的记录: {len(complete2)}")
print(f"唯一股票代码: {df2['股票代码'].nunique()}")

# Notes
notes2 = df2[df2['备注'].notna()]
print(f"\n有备注的记录: {len(notes2)}")
for i, row in notes2.iterrows():
    print(f"  [{i}] {row['股票名称']}: {row['备注']}")

# Check other sheets
print()
for sname in xl2.sheet_names[1:]:
    dfs = pd.read_excel(path2, sheet_name=sname)
    print(f"Sheet '{sname}': {dfs.shape[0]}行 x {dfs.shape[1]}列, 列={list(dfs.columns)[:5]}")

print()
print("=" * 80)
print("与当前实验的23个案例对比")
print("=" * 80)

# Current 23 cases from thesis - check overlap
# Let's see what the file2 additional sheets contain
print("\n文件2中的 '全部股票代码（手动整理版）' Sheet:")
df_codes = pd.read_excel(path2, sheet_name='全部股票代码（手动整理版）')
print(df_codes.head(10).to_string())
print(f"共 {len(df_codes)} 个代码")

# Summary analysis
print()
print("=" * 80)
print("可用性分析")
print("=" * 80)

# File 1 analysis
valid1 = df1[df1['股票代码'].notna() & df1['开始日期'].notna() & df1['结束日期'].notna()].copy()
valid1['duration'] = (valid1['结束日期'] - valid1['开始日期']).dt.days

# Filter: only A-share (SH/SZ), duration >= 30 days
ashare1 = valid1[valid1['股票代码'].str.contains(r'\.(SH|SZ)$', na=False)]
print(f"\n文件1 - A股(SH/SZ)且有完整日期: {len(ashare1)}")
long_enough1 = ashare1[ashare1['duration'] >= 30]
print(f"文件1 - 其中持续>=30天: {len(long_enough1)}")
long_enough1_60 = ashare1[ashare1['duration'] >= 60]
print(f"文件1 - 其中持续>=60天: {len(long_enough1_60)}")

# Show sample of usable cases
print(f"\n持续>=60天的案例示例(前20):")
for i, row in long_enough1_60.head(20).iterrows():
    print(f"  {row['股票代码']} {row['股票名称']} | {str(row['开始日期'])[:10]}~{str(row['结束日期'])[:10]} | {row['duration']}天 | 处罚:{str(row['处罚宣告日'])[:10]}")

# Check how many unique stocks with enough duration
print(f"\n持续>=60天的唯一股票: {long_enough1_60['股票代码'].nunique()}")
print(f"持续>=30天的唯一股票: {long_enough1['股票代码'].nunique()}")
