"""Re-upload all fixed files and run Round 1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import upload_file, remote_exec

model_dir = Path(__file__).parent / "model"

# Re-upload all changed files
files_to_upload = [
    ("configs/pretrain_config.yaml", "/root/project/model/configs/pretrain_config.yaml"),
    ("data/crypto_dataset.py", "/root/project/model/data/crypto_dataset.py"),
    ("scripts/run_pretrain.py", "/root/project/model/scripts/run_pretrain.py"),
]
for local_rel, remote_path in files_to_upload:
    upload_file(str(model_dir / local_rel), remote_path)
    print(f"  Uploaded {local_rel}")

print("\nLaunching Round 1 pre-training...")
print("="*60)
code, out, err = remote_exec(
    "source /etc/profile && "
    "cd /root/project/model && "
    "python -m scripts.run_pretrain "
    "--config configs/pretrain_config.yaml "
    "--round 1 "
    "2>&1",
    timeout=7200,
    print_output=True,
)
print(f"\nExit code: {code}")
