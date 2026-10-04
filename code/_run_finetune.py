"""
Phase 3 local execution: download pretrained weights + run fine-tuning.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
DATA_DIR = BASE / "data_pipeline" / "data"

os.chdir(str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR))


def install_deps():
    """Install required packages for fine-tuning."""
    core_deps = ["torch", "numpy", "pandas", "pyarrow", "pyyaml", "loguru"]
    for dep in core_deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])


def download_pretrained_weights():
    """Download best.pt from AutoDL via paramiko."""
    local_path = MODEL_DIR / "outputs" / "pretrain" / "round3_full" / "best.pt"
    if local_path.exists():
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"Pre-trained weights already exist: {local_path} ({size_mb:.1f} MB)")
        return str(local_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading pre-trained weights from AutoDL...")
    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname="connect.westb.seetacloud.com",
            port=49397,
            username="root",
            password=os.environ["AUTODL_SSH_PASSWORD"],
        )
        sftp = ssh.open_sftp()
        remote_path = "/root/project/outputs/pretrain/round3_full/best.pt"
        sftp.get(remote_path, str(local_path))
        sftp.close()
        ssh.close()
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"Downloaded: {local_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Will proceed without pre-trained weights (random init)")
        return ""

    return str(local_path)


def prepare_data_paths(cfg: dict) -> dict:
    """Adjust config paths to local directories."""
    cfg["data"]["cases_dir"] = str(DATA_DIR / "processed" / "extracted_cases")
    cfg["data"]["astock_dir"] = str(DATA_DIR / "raw" / "astock")
    cfg["data"]["cot_results"] = str(MODEL_DIR / "outputs" / "cot" / "cot_results.json")
    return cfg


def run_level(level: str, cfg: dict):
    """Run fine-tuning for a specific level."""
    import yaml

    temp_cfg_path = MODEL_DIR / "configs" / "_finetune_local.yaml"
    with open(temp_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    cmd = [
        sys.executable, "-m", "scripts.run_finetune",
        "--config", str(temp_cfg_path),
        "--level", level,
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(MODEL_DIR))
    return result.returncode


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="all", choices=["L0", "L1", "L2", "L3", "all"])
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    install_deps()

    import yaml
    cfg_path = MODEL_DIR / "configs" / "finetune_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not args.skip_download:
        ckpt_path = download_pretrained_weights()
        if ckpt_path:
            cfg["pretrained"]["checkpoint"] = ckpt_path

    cfg = prepare_data_paths(cfg)
    cfg["output"]["dir"] = str(MODEL_DIR / "outputs" / "finetune")

    rc = run_level(args.level, cfg)
    print(f"\nExit code: {rc}")
