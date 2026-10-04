# -*- coding: utf-8 -*-
"""
从两个第三方Excel文件 + 现有23个CSRC案例，构建统一的操纵案例数据集。
筛选条件：沪深A股(SH/SZ)、有完整操纵起止日期、持续>=60自然日。
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.stdout.reconfigure(encoding='utf-8')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 80)

BASE = Path(r'c:\Users\wangm\OneDrive - Audencia\毕业论文')
PATH1 = BASE / '操作处罚案例2012-2024.xlsx'
PATH2 = BASE / '2024年以后的操作处罚案例记录.xlsx'
ASTOCK_RAW = BASE / 'code' / 'data_pipeline' / 'data' / 'raw' / 'astock'
EXTRACTED = BASE / 'code' / 'data_pipeline' / 'data' / 'processed' / 'extracted_cases'
OUTPUT = BASE / 'manipulation_cases_dataset.csv'

# ── helpers ──────────────────────────────────────────────────────────────────
def parse_mixed_dates(series):
    result = pd.Series(pd.NaT, index=series.index)
    for i, val in series.items():
        if pd.isna(val):
            continue
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            result[i] = pd.Timestamp(val)
        elif isinstance(val, str):
            try:
                result[i] = pd.to_datetime(val)
            except Exception:
                pass
        elif isinstance(val, (int, float)):
            try:
                result[i] = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(val))
            except Exception:
                pass
    return result

def normalize_code(code_str):
    """Normalize stock code to 6-digit + .SH/.SZ format."""
    if pd.isna(code_str):
        return None
    s = str(code_str).strip()
    if s.endswith('.SH') or s.endswith('.SZ'):
        return s
    return None

NON_MANIP_KEYWORDS = [
    '并非操纵', '信息披露', '期货合约', '操纵基金', '从业人员违法',
    '保荐未尽责', '超比例持有', '公司信息造假', '未按规定披露',
    '未公开发行', '操纵债券', '操纵期货'
]

def is_non_manipulation(note):
    if pd.isna(note):
        return False
    return any(kw in str(note) for kw in NON_MANIP_KEYWORDS)

# ── 1. Load existing 23 cases ───────────────────────────────────────────────
print("=" * 80)
print("Step 1: 读取现有23个CSRC案例")
print("=" * 80)

existing_cases = []
for case_dir in sorted(ASTOCK_RAW.iterdir()):
    if not case_dir.is_dir():
        continue
    case_id = case_dir.name
    ext_file = EXTRACTED / f'{case_id}_extracted.json'
    if ext_file.exists():
        with open(ext_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cases_list = data.get('cases', [data]) if isinstance(data, dict) else data
        if isinstance(cases_list, dict):
            cases_list = [cases_list]
        for c in cases_list:
            code_raw = c.get('stock_code', '')
            name = c.get('stock_name', c.get('company_name', ''))
            code6 = str(code_raw).zfill(6) if code_raw else ''
            if code6.startswith('6'):
                code_full = code6 + '.SH'
            elif code6.startswith(('0', '3')):
                code_full = code6 + '.SZ'
            else:
                code_full = code6
            existing_cases.append({
                'case_id': case_id,
                'stock_code': code_full,
                'stock_name': name,
                'source': 'CSRC_direct'
            })

existing_codes = set(c['stock_code'] for c in existing_cases)
print(f"现有案例: {len(existing_cases)} 个")
for c in existing_cases:
    print(f"  {c['case_id']:20s} | {c['stock_code']} {c['stock_name']}")

# ── 2. Load & filter Excel file 2 (superset) ────────────────────────────────
print()
print("=" * 80)
print("Step 2: 从文件2(超集)读取并筛选")
print("=" * 80)

df = pd.read_excel(PATH2, sheet_name=0)
print(f"原始行数: {len(df)}")

df['stock_code_norm'] = df['股票代码'].apply(normalize_code)
df['start_date'] = parse_mixed_dates(df['开始日期'])
df['end_date'] = parse_mixed_dates(df['结束日期'])
df['penalty_date'] = parse_mixed_dates(df['处罚宣告日'])
df['is_non_manip'] = df['备注'].apply(is_non_manipulation)

# Filter step by step
mask_code = df['stock_code_norm'].notna()
mask_dates = df['start_date'].notna() & df['end_date'].notna()
mask_ashare = df['stock_code_norm'].str.match(r'^\d{6}\.(SH|SZ)$', na=False)
mask_manip = ~df['is_non_manip']

df_valid = df[mask_code & mask_dates & mask_ashare & mask_manip].copy()
df_valid['duration'] = (df_valid['end_date'] - df_valid['start_date']).dt.days
print(f"有代码+日期+A股+操纵: {len(df_valid)}")

# Duration >= 60
df_long = df_valid[df_valid['duration'] >= 60].copy()
print(f"持续>=60天: {len(df_long)}, 唯一股票: {df_long['stock_code_norm'].nunique()}")

# Also check file1 for any additional cases
df1 = pd.read_excel(PATH1, sheet_name='Sheet1')
df1['stock_code_norm'] = df1['股票代码'].apply(normalize_code)
df1['start_date'] = parse_mixed_dates(df1['开始日期'])
df1['end_date'] = parse_mixed_dates(df1['结束日期'])
df1['penalty_date'] = parse_mixed_dates(df1['处罚宣告日'])
df1_valid = df1[df1['stock_code_norm'].notna() & df1['start_date'].notna() & df1['end_date'].notna()].copy()
df1_valid = df1_valid[df1_valid['stock_code_norm'].str.match(r'^\d{6}\.(SH|SZ)$', na=False)]
df1_valid['duration'] = (df1_valid['end_date'] - df1_valid['start_date']).dt.days
df1_long = df1_valid[df1_valid['duration'] >= 60]

# Merge: use file2 as primary, add file1 extras
codes_in_f2 = set(zip(df_long['stock_code_norm'], df_long['start_date'].dt.date))
extra_from_f1 = []
for _, row in df1_long.iterrows():
    key = (row['stock_code_norm'], row['start_date'].date())
    if key not in codes_in_f2:
        extra_from_f1.append(row)
print(f"文件1中独有的>=60天案例: {len(extra_from_f1)}")

# ── 3. Build unified dataset ────────────────────────────────────────────────
print()
print("=" * 80)
print("Step 3: 构建统一数据集")
print("=" * 80)

records = []
record_id = 0

# 3a: From Excel (file2 primary)
for _, row in df_long.iterrows():
    record_id += 1
    note = str(row['备注']) if pd.notna(row['备注']) else ''
    in_existing = row['stock_code_norm'] in existing_codes
    records.append({
        'id': record_id,
        'stock_code': row['stock_code_norm'],
        'stock_name': row['股票名称'] if pd.notna(row['股票名称']) else '',
        'start_date': row['start_date'].strftime('%Y-%m-%d'),
        'end_date': row['end_date'].strftime('%Y-%m-%d'),
        'duration_days': int(row['duration']),
        'penalty_date': row['penalty_date'].strftime('%Y-%m-%d') if pd.notna(row['penalty_date']) else '',
        'penalty_id': str(row['文本序']) if pd.notna(row['文本序']) else '',
        'note': note,
        'source': 'third_party_excel',
        'overlap_with_existing': in_existing,
    })

# 3b: From file1 extras
for row in extra_from_f1:
    record_id += 1
    note = str(row['备注']) if pd.notna(row['备注']) else ''
    in_existing = row['stock_code_norm'] in existing_codes
    records.append({
        'id': record_id,
        'stock_code': row['stock_code_norm'],
        'stock_name': row['股票名称'] if pd.notna(row['股票名称']) else '',
        'start_date': row['start_date'].strftime('%Y-%m-%d'),
        'end_date': row['end_date'].strftime('%Y-%m-%d'),
        'duration_days': int(row['duration']),
        'penalty_date': row['penalty_date'].strftime('%Y-%m-%d') if pd.notna(row['penalty_date']) else '',
        'penalty_id': str(row['文本序']) if pd.notna(row['文本序']) else '',
        'note': note,
        'source': 'third_party_excel_f1_extra',
        'overlap_with_existing': in_existing,
    })

df_all = pd.DataFrame(records)

# 3c: Add existing 23 cases that are NOT in the Excel
# (These are from CSRC direct, might have different date info)
existing_not_in_excel = []
excel_codes = set(df_all['stock_code'].unique())
for c in existing_cases:
    if c['stock_code'] not in excel_codes:
        existing_not_in_excel.append(c)
        record_id += 1
        df_all = pd.concat([df_all, pd.DataFrame([{
            'id': record_id,
            'stock_code': c['stock_code'],
            'stock_name': c['stock_name'],
            'start_date': '',
            'end_date': '',
            'duration_days': -1,
            'penalty_date': '',
            'penalty_id': c['case_id'],
            'note': '来自CSRC直接爬取,Excel中无此案例',
            'source': 'CSRC_direct_only',
            'overlap_with_existing': True,
        }])], ignore_index=True)

print(f"Excel筛选(>=60天): {len(df_long)} 条")
print(f"文件1补充: {len(extra_from_f1)} 条")
print(f"现有23案例中Excel未覆盖: {len(existing_not_in_excel)} 条")
print(f"合计记录: {len(df_all)} 条")
print(f"唯一股票代码: {df_all['stock_code'].nunique()}")

# ── 4. Deduplicate ──────────────────────────────────────────────────────────
print()
print("=" * 80)
print("Step 4: 去重处理")
print("=" * 80)

# Same stock + overlapping date ranges → keep the longer one
df_excel = df_all[df_all['source'] != 'CSRC_direct_only'].copy()
df_csrc_only = df_all[df_all['source'] == 'CSRC_direct_only'].copy()

# For excel records: deduplicate by stock_code + start_date
df_excel_dedup = df_excel.drop_duplicates(subset=['stock_code', 'start_date'], keep='first')
print(f"Excel去重前: {len(df_excel)}, 去重后: {len(df_excel_dedup)}")

df_final = pd.concat([df_excel_dedup, df_csrc_only], ignore_index=True)
df_final = df_final.sort_values(['stock_code', 'start_date']).reset_index(drop=True)
df_final['id'] = range(1, len(df_final) + 1)

print(f"最终数据集: {len(df_final)} 条记录, {df_final['stock_code'].nunique()} 只股票")

# ── 5. Save ─────────────────────────────────────────────────────────────────
df_final.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f"\n已保存至: {OUTPUT}")

# ── 6. Summary statistics ───────────────────────────────────────────────────
print()
print("=" * 80)
print("Step 5: 数据集统计")
print("=" * 80)

df_with_dur = df_final[df_final['duration_days'] >= 0]
print(f"\n有完整日期的记录: {len(df_with_dur)}")
print(f"CSRC直接来源(无Excel日期): {len(df_csrc_only)}")
print(f"\n持续天数统计(有日期的记录):")
print(f"  均值: {df_with_dur['duration_days'].mean():.1f} 天")
print(f"  中位数: {df_with_dur['duration_days'].median():.1f} 天")
print(f"  最小: {df_with_dur['duration_days'].min()} 天")
print(f"  最大: {df_with_dur['duration_days'].max()} 天")

# Duration distribution
bins = [60, 90, 120, 180, 365, 1000, 99999]
labels = ['60-90天', '90-120天', '120-180天', '180-365天', '365-1000天', '>1000天']
df_with_dur['dur_bin'] = pd.cut(df_with_dur['duration_days'], bins=bins, labels=labels, right=False)
print(f"\n持续天数分布:")
print(df_with_dur['dur_bin'].value_counts().sort_index().to_string())

# Overlap analysis
overlap = df_final[df_final['overlap_with_existing'] == True]
new_only = df_final[df_final['overlap_with_existing'] == False]
print(f"\n与现有23案例重叠: {len(overlap)} 条 ({overlap['stock_code'].nunique()} 只股票)")
print(f"全新案例: {len(new_only)} 条 ({new_only['stock_code'].nunique()} 只股票)")

# Print full dataset
print()
print("=" * 80)
print("完整案例列表")
print("=" * 80)
for _, row in df_final.iterrows():
    overlap_mark = '*' if row['overlap_with_existing'] else ' '
    note_short = row['note'][:30] if row['note'] else ''
    dur = f"{row['duration_days']:4d}天" if row['duration_days'] >= 0 else ' N/A '
    print(f"  {row['id']:3d} {overlap_mark} {row['stock_code']} {row['stock_name']:8s} | "
          f"{row['start_date']:10s} ~ {row['end_date']:10s} | {dur} | {row['source']:25s} | {note_short}")

print(f"\n* = 与现有23个CSRC案例的股票代码重叠")
print(f"\n最终数据集规模: {len(df_final)} 条记录, {df_final['stock_code'].nunique()} 只不同股票")
