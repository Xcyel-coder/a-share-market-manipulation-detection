"""Check Round 3 training status."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec

print("--- GPU & Process ---")
remote_exec("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null")
remote_exec("ps aux | grep run_pretrain | grep -v grep || echo 'No training process'")

print("\n--- Round 3 Checkpoints ---")
remote_exec("ls -lh /root/project/outputs/pretrain/round3_full/ 2>/dev/null || echo 'No round3 directory'")

print("\n--- Last 25 lines of log ---")
remote_exec("tail -25 /root/project/outputs/pretrain/pretrain.log 2>/dev/null || echo 'No log'")
