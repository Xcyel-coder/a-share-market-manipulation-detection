"""Upload fixed config and run Round 2 (50% data)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import upload_file, remote_exec

model_dir = Path(__file__).parent / "model"

files = [
    ("configs/pretrain_config.yaml", "/root/project/model/configs/pretrain_config.yaml"),
    ("data/crypto_dataset.py", "/root/project/model/data/crypto_dataset.py"),
    ("data/feature_engineering.py", "/root/project/model/data/feature_engineering.py"),
]
for local_rel, remote_path in files:
    upload_file(str(model_dir / local_rel), remote_path)
    print(f"  Uploaded {local_rel}")

print("\nLaunching Round 2 (50% data, 25 epochs, batch=256)...")
print("="*60)
code, out, err = remote_exec(
    "source /etc/profile && "
    "cd /root/project/model && "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "python -m scripts.run_pretrain "
    "--config configs/pretrain_config.yaml "
    "--round 2 "
    "--resume /root/project/outputs/pretrain/round1_validation/best.pt "
    "2>&1",
    timeout=7200,
    print_output=True,
)
print(f"\nExit code: {code}")
