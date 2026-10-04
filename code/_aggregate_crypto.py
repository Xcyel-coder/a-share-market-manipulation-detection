# -*- coding: utf-8 -*-
"""
在AutoDL服务器上执行：将加密货币分钟级数据聚合为日线和4小时OHLCV。
本脚本通过SSH上传并在远端执行。
"""
import paramiko
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

REMOTE_SCRIPT = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate crypto minute data to daily and 4h OHLCV."""
import os
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("/root/project/data/raw/crypto")
OUT_DIR = Path("/root/project/data/processed/crypto_aggregated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT"]

def load_symbol(symbol):
    """Load all parquet files for a symbol and concatenate."""
    sym_dir = RAW_DIR / symbol
    if not sym_dir.exists():
        print(f"  [SKIP] {symbol}: directory not found")
        return None
    dfs = []
    for f in sorted(sym_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df)} rows")
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df

def aggregate_to_timeframe(df, freq, symbol):
    """Aggregate minute OHLCV to target frequency."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    elif "datetime" not in df.columns:
        raise ValueError("No timestamp or datetime column")

    df = df.set_index("datetime")
    
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "quote_volume" in df.columns:
        agg_dict["quote_volume"] = "sum"
    if "trades" in df.columns:
        agg_dict["trades"] = "sum"
    
    resampled = df.resample(freq).agg(agg_dict).dropna(subset=["open"])
    resampled = resampled.reset_index()
    resampled["timestamp"] = resampled["datetime"].astype(np.int64) // 10**6
    
    return resampled

print("=" * 60)
print("Aggregating crypto minute data to daily and 4h")
print("=" * 60)

for symbol in SYMBOLS:
    print(f"\n--- {symbol} ---")
    df = load_symbol(symbol)
    if df is None:
        continue
    print(f"  Total minute bars: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {pd.to_datetime(df['timestamp'].min(), unit='ms')} ~ {pd.to_datetime(df['timestamp'].max(), unit='ms')}")
    
    # Aggregate to daily
    daily = aggregate_to_timeframe(df, "1D", symbol)
    daily_path = OUT_DIR / f"{symbol}_daily.parquet"
    daily.to_parquet(daily_path, index=False)
    print(f"  Daily: {len(daily)} bars -> {daily_path}")
    
    # Aggregate to 4 hours
    h4 = aggregate_to_timeframe(df, "4h", symbol)
    h4_path = OUT_DIR / f"{symbol}_4h.parquet"
    h4.to_parquet(h4_path, index=False)
    print(f"  4H: {len(h4)} bars -> {h4_path}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for f in sorted(OUT_DIR.glob("*.parquet")):
    df = pd.read_parquet(f)
    print(f"  {f.name}: {len(df)} bars, cols={list(df.columns)[:6]}")

# Also check a sample daily bar
sample = pd.read_parquet(OUT_DIR / "BTCUSDT_daily.parquet")
print(f"\nBTCUSDT daily sample (first 5 rows):")
print(sample.head().to_string())
print(f"\nBTCUSDT daily sample (last 5 rows):")
print(sample.tail().to_string())

print("\nDone!")
'''

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Find python
stdin, stdout, stderr = ssh.exec_command("which python3 || which python || find /root -name 'python3' -type f 2>/dev/null | head -5 || find /opt -name 'python3' -type f 2>/dev/null | head -5", timeout=15)
py_paths = stdout.read().decode().strip()
print(f"Python paths found: {py_paths}")

# Try conda
stdin, stdout, stderr = ssh.exec_command("source /etc/profile && conda info --envs 2>/dev/null; ls /root/miniconda3/bin/python* 2>/dev/null; ls /opt/conda/bin/python* 2>/dev/null", timeout=15)
conda_out = stdout.read().decode().strip()
print(f"Conda info: {conda_out}")

# Upload script
sftp = ssh.open_sftp()
with sftp.open("/root/project/aggregate_crypto.py", "w") as f:
    f.write(REMOTE_SCRIPT)
sftp.close()
print("Uploaded aggregate_crypto.py")

# Find working python and execute
python_candidates = [
    "/root/miniconda3/bin/python",
    "/opt/conda/bin/python",
    "/usr/bin/python3",
    "python3",
    "python",
]

for py in python_candidates:
    stdin, stdout, stderr = ssh.exec_command(f"{py} --version 2>&1", timeout=10)
    ver = stdout.read().decode().strip()
    if "Python" in ver:
        print(f"Using: {py} ({ver})")
        # Run the script
        print(f"Running aggregation...")
        stdin, stdout, stderr = ssh.exec_command(
            f"cd /root/project && {py} aggregate_crypto.py 2>&1",
            timeout=300
        )
        output = stdout.read().decode()
        print(output)
        err = stderr.read().decode()
        if err.strip():
            print(f"[STDERR] {err[:500]}")
        break
else:
    print("ERROR: No working Python found!")

ssh.close()
print("Script finished.")
