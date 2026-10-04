"""Check total data size per symbol on remote."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import upload_file, remote_exec

# Create a remote script
script = '''
import pandas as pd, os
base = "/root/project/data/raw/crypto"
grand_total = 0
for sym in sorted(os.listdir(base)):
    d = os.path.join(base, sym)
    if not os.path.isdir(d):
        continue
    total = 0
    for f in sorted(os.listdir(d)):
        fp = os.path.join(d, f)
        df = pd.read_parquet(fp)
        total += len(df)
        print(f"  {sym}/{f}: {len(df):>10,} rows")
    print(f"  {sym} TOTAL: {total:>10,}")
    print()
    grand_total += total
print(f"GRAND TOTAL: {grand_total:>10,} rows")
'''

# Write and run on remote
remote_exec(f"cat << 'PYEOF' > /tmp/check_data.py\n{script}\nPYEOF")
remote_exec("source /etc/profile && python /tmp/check_data.py")
