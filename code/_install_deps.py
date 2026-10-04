"""Step 1: Install dependencies on remote."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec

print("Installing Python packages on remote GPU (this takes 2-3 min)...")
code, out, err = remote_exec(
    "source /etc/profile && "
    "python -m pip install -q "
    "transformers accelerate peft safetensors datasets "
    "pandas numpy pyarrow scikit-learn xgboost "
    "ta wandb loguru tensorboard "
    "pyyaml tqdm matplotlib seaborn "
    "&& echo INSTALL_OK",
    timeout=600,
    print_output=True,
)
if "INSTALL_OK" in out:
    print("\n=== Dependencies installed successfully ===")
else:
    print(f"\n=== Install may have issues (exit code: {code}) ===")
