"""Final verification of remote environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec

print("="*60)
print("REMOTE ENVIRONMENT VERIFICATION")
print("="*60)

remote_exec(
    "source /etc/profile && "
    "echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
    "echo '=== Python Packages ===' && "
    "python -c \""
    "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}');"
    "import transformers; print(f'  Transformers: {transformers.__version__}');"
    "import pandas; print(f'  Pandas: {pandas.__version__}');"
    "import sklearn; print(f'  Scikit-learn: {sklearn.__version__}');"
    "import xgboost; print(f'  XGBoost: {xgboost.__version__}');"
    "import loguru; print(f'  Loguru: OK');"
    "import yaml; print(f'  PyYAML: OK');"
    "import ta; print(f'  TA-Lib: OK');"
    "print('  All packages OK!');"
    "\" && "
    "echo '=== Data ===' && "
    "find /root/project/data/raw/crypto -mindepth 1 -maxdepth 1 -type d -exec sh -c 'echo \"  {} : $(find {} -name *.parquet | wc -l) files\"' \\; && "
    "echo '=== Code ===' && "
    "find /root/project/model -name '*.py' -exec echo '  {}' \\; && "
    "echo '=== READY FOR PRE-TRAINING ==='",
    timeout=30,
)
