"""Upload baseline code and run on remote."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import upload_file, upload_dir, remote_exec

model_dir = Path(__file__).parent / "model"

# Upload baseline module
upload_dir(str(model_dir / "baselines"), "/root/project/model/baselines")
upload_file(str(model_dir / "scripts" / "run_baselines.py"), "/root/project/model/scripts/run_baselines.py")

print("\nRunning baseline experiments (RF + XGBoost)...")
print("=" * 60)
code, out, err = remote_exec(
    "source /etc/profile && "
    "cd /root/project/model && "
    "python -m scripts.run_baselines "
    "--config configs/pretrain_config.yaml "
    "2>&1",
    timeout=3600,
    print_output=True,
)
print(f"\nExit code: {code}")
