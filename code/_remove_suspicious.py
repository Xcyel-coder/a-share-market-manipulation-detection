# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

CSV = r"c:\Users\wangm\OneDrive - Audencia\毕业论文\manipulation_cases_verified.csv"
df = pd.read_csv(CSV, encoding="utf-8-sig")
print(f"原始记录: {len(df)}")

sus = df[df["duration_days"] > 1000]
print(f"\n剔除的超长案例 ({len(sus)} 个):")
for _, r in sus.iterrows():
    print(f"  id={int(r['id']):3d}  {r['stock_code']}  {r['stock_name']:<8s}  {int(r['duration_days'])}天  [{r['quality_tier']}]")

df2 = df[df["duration_days"] <= 1000].copy()
print(f"\n剔除后记录: {len(df2)}")

for tier in ["Gold", "Silver", "Bronze"]:
    n = len(df2[df2["quality_tier"] == tier])
    print(f"  {tier}: {n}")

df2.to_csv(CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存至 {CSV}")
