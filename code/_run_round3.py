"""Run Round 3 (full data, 40 epochs), resuming from Round 2 best checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import upload_file, remote_exec

model_dir = Path(__file__).parent / "model"

# Re-upload config (in case of any local edits)
upload_file(
    str(model_dir / "configs" / "pretrain_config.yaml"),
    "/root/project/model/configs/pretrain_config.yaml",
)

print("\nLaunching Round 3 (100% data, 40 epochs)...")
print("=" * 60)
code, out, err = remote_exec(
    "source /etc/profile && "
    "cd /root/project/model && "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "python -m scripts.run_pretrain "
    "--config configs/pretrain_config.yaml "
    "--round 3 "
    "--resume /root/project/outputs/pretrain/round2_tuning/best.pt "
    "2>&1",
    timeout=14400,  # 4 hours max
    print_output=True,
)
print(f"\nExit code: {code}")
