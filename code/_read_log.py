import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec
remote_exec("tail -20 /root/project/outputs/pretrain/pretrain.log 2>/dev/null || echo 'No log yet'")
