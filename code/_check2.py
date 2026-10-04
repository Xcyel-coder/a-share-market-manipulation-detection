import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec
remote_exec(
    "ls -la /root/project/crypto_data.tar.gz 2>/dev/null && echo 'UPLOADING...' || echo 'NO_TARBALL' ; "
    "find /root/project/data/raw/crypto -name '*.parquet' 2>/dev/null | wc -l ; "
    "du -sh /root/project/data/ 2>/dev/null"
)
