"""Upload crypto data as compressed tar archive (much faster than SFTP file-by-file)."""
import sys
import os
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from remote import upload_file, remote_exec

LOCAL_CRYPTO = Path(__file__).parent / "data_pipeline" / "data" / "raw" / "crypto"
REMOTE_ROOT = "/root/project"

if not LOCAL_CRYPTO.exists():
    print(f"ERROR: {LOCAL_CRYPTO} not found")
    sys.exit(1)

# Count local files
local_files = list(LOCAL_CRYPTO.rglob("*.parquet"))
print(f"Found {len(local_files)} parquet files locally")

# Create tar.gz
tar_path = os.path.join(tempfile.gettempdir(), "crypto_data.tar.gz")
print(f"Compressing to {tar_path}...")
with tarfile.open(tar_path, "w:gz") as tar:
    for pf in local_files:
        arcname = str(pf.relative_to(LOCAL_CRYPTO.parent.parent))
        tar.add(str(pf), arcname=arcname)
        
size_mb = os.path.getsize(tar_path) / (1024 * 1024)
print(f"Archive size: {size_mb:.1f} MB")

# Upload
print(f"\nUploading archive to remote...")
upload_file(tar_path, f"{REMOTE_ROOT}/crypto_data.tar.gz")
print("Upload complete!")

# Extract on remote
print("\nExtracting on remote...")
remote_exec(
    f"cd {REMOTE_ROOT}/data/raw && "
    f"tar xzf {REMOTE_ROOT}/crypto_data.tar.gz && "
    f"rm {REMOTE_ROOT}/crypto_data.tar.gz && "
    f"echo EXTRACTED && "
    "find crypto -name '*.parquet' | wc -l && "
    "du -sh crypto/"
)
print("\nDone!")

# Cleanup local temp
os.remove(tar_path)
