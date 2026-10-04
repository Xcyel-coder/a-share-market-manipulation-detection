"""Check remote state and run missing steps individually."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec

print("=== FULL REMOTE STATUS ===")
remote_exec(
    "source /etc/profile && "
    "echo '--- Python packages (ML) ---' && "
    "pip list 2>/dev/null | grep -iE 'torch|transformers|accelerate|peft|bitsandbytes|scikit|xgboost|loguru|pandas|ta ' && "
    "echo '--- Project tree ---' && "
    "find /root/project -type f 2>/dev/null | head -30 && "
    "echo '--- Parquet count ---' && "
    "find /root/project/data -name '*.parquet' 2>/dev/null | wc -l && "
    "echo '--- Disk ---' && "
    "du -sh /root/project/* 2>/dev/null && "
    "echo DONE"
)
