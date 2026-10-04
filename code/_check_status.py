"""Check full remote training status."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec

print("=" * 60)
print("REMOTE TRAINING STATUS CHECK")
print("=" * 60)

print("\n--- GPU & Running Processes ---")
remote_exec("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null")
remote_exec("ps aux | grep run_pretrain | grep -v grep || echo 'No training process running'")

print("\n--- Training Checkpoints ---")
remote_exec("find /root/project/outputs/pretrain -name '*.pt' -exec ls -lh {} \\; 2>/dev/null | sort || echo 'No checkpoints'")

print("\n--- Pretrain Summary ---")
remote_exec("cat /root/project/outputs/pretrain/pretrain_summary.json 2>/dev/null || echo 'No summary'")

print("\n--- Last 30 lines of training log ---")
remote_exec("tail -30 /root/project/outputs/pretrain/pretrain.log 2>/dev/null || echo 'No log'")
