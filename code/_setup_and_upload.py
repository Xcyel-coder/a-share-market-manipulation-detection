"""
One-time setup: install deps on remote, upload code and crypto data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from remote import remote_exec, upload_dir, upload_file

REMOTE_ROOT = "/root/project"


def step1_install_deps():
    print("\n" + "=" * 60)
    print("STEP 1: Installing dependencies on remote GPU")
    print("=" * 60)
    setup_script = Path(__file__).parent / "model" / "scripts" / "setup_remote.sh"
    upload_file(str(setup_script), f"{REMOTE_ROOT}/setup_remote.sh")
    code, _, _ = remote_exec(f"source /etc/profile && bash {REMOTE_ROOT}/setup_remote.sh", timeout=600)
    if code != 0:
        print("WARNING: Setup had errors, check output above")
    return code


def step2_upload_code():
    print("\n" + "=" * 60)
    print("STEP 2: Uploading model code")
    print("=" * 60)
    local_model_dir = Path(__file__).parent / "model"
    upload_dir(str(local_model_dir), f"{REMOTE_ROOT}/model")


def step3_upload_crypto_data():
    print("\n" + "=" * 60)
    print("STEP 3: Uploading crypto data (this may take a few minutes)")
    print("=" * 60)
    local_crypto = Path(__file__).parent / "data_pipeline" / "data" / "raw" / "crypto"
    if not local_crypto.exists():
        print(f"WARNING: Crypto data not found at {local_crypto}")
        print("Skipping data upload. Upload manually later.")
        return

    for symbol_dir in sorted(local_crypto.iterdir()):
        if symbol_dir.is_dir():
            remote_symbol = f"{REMOTE_ROOT}/data/raw/crypto/{symbol_dir.name}"
            print(f"\nUploading {symbol_dir.name}...")
            upload_dir(str(symbol_dir), remote_symbol)


def step4_verify():
    print("\n" + "=" * 60)
    print("STEP 4: Verifying remote setup")
    print("=" * 60)
    remote_exec(
        f"source /etc/profile && cd {REMOTE_ROOT} && "
        "echo '--- Code files ---' && "
        "find model/ -name '*.py' | head -20 && "
        "echo '--- Data files ---' && "
        "find data/raw/crypto -name '*.parquet' 2>/dev/null | wc -l && "
        "echo '--- Disk usage ---' && "
        "du -sh data/ model/ 2>/dev/null"
    )


if __name__ == "__main__":
    step1_install_deps()
    step2_upload_code()
    step3_upload_crypto_data()
    step4_verify()
    print("\n" + "=" * 60)
    print("SETUP COMPLETE! Ready for pre-training.")
    print("=" * 60)
