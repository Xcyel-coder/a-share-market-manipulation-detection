"""Launch Round 1 pre-training on remote GPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec

print("Launching Round 1 pre-training (10% data, 15 epochs)...")
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
