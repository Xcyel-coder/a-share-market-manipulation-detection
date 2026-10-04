import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec
remote_exec(
    "source /etc/profile && "
    "echo DEPS: && pip list 2>/dev/null | grep -cE 'transformers|scikit-learn|xgboost|loguru' && "
    "echo PARQUETS: && find /root/project/data/raw/crypto -name '*.parquet' 2>/dev/null | wc -l && "
    "echo DISK: && du -sh /root/project/data/ 2>/dev/null"
)
