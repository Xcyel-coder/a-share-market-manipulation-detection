"""Quick check: is remote setup done?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec

print("--- Checking dependencies ---")
remote_exec("source /etc/profile && python -c 'import transformers; print(\"transformers:\", transformers.__version__)' 2>&1 || echo 'NOT INSTALLED'")

print("\n--- Checking data ---")
remote_exec("source /etc/profile && find /root/project/data/raw/crypto -name '*.parquet' 2>/dev/null | wc -l")

print("\n--- Checking code ---")
remote_exec("source /etc/profile && ls /root/project/model/models/ 2>/dev/null || echo 'NO CODE'")
