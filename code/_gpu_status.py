import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec
remote_exec("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null && ps aux | grep run_pretrain | grep -v grep")
