"""Step 2: Upload remaining crypto data to remote."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import upload_dir, remote_exec

REMOTE_ROOT = "/root/project"
local_crypto = Path(__file__).parent / "data_pipeline" / "data" / "raw" / "crypto"

if not local_crypto.exists():
    print(f"ERROR: {local_crypto} not found")
    sys.exit(1)

for symbol_dir in sorted(local_crypto.iterdir()):
    if not symbol_dir.is_dir():
        continue
    remote_path = f"{REMOTE_ROOT}/data/raw/crypto/{symbol_dir.name}"
    n_local = len(list(symbol_dir.glob("*.parquet")))
    print(f"\nUploading {symbol_dir.name} ({n_local} files)...")
    upload_dir(str(symbol_dir), remote_path)

print("\n--- Verifying ---")
remote_exec(
    "find /root/project/data/raw/crypto -name '*.parquet' | wc -l && "
    "du -sh /root/project/data/raw/crypto/"
)
print("Data upload complete!")
