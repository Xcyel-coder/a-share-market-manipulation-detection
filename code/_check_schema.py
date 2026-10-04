"""Check parquet schema on remote."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec

remote_exec(
    "source /etc/profile && python -c \""
    "import pandas as pd; "
    "import os; "
    "d='/root/project/data/raw/crypto/BTCUSDT'; "
    "f=sorted(os.listdir(d))[0]; "
    "df=pd.read_parquet(os.path.join(d,f)); "
    "print('Columns:', list(df.columns)); "
    "print('Shape:', df.shape); "
    "print('dtypes:'); print(df.dtypes); "
    "print('Head:'); print(df.head(3)); "
    "\""
)
