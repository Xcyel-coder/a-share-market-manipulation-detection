"""Verify code fixes and all result files on the new instance."""
import os
import paramiko
import json

HOST = "connect.westd.seetacloud.com"
PORT = 28784
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)


def run(cmd):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    return stdout.read().decode().strip()


# 1. Check code fixes
print("=" * 60)
print("1. Verify metadata routing fix in finetune_model.py")
print("=" * 60)
out = run("grep -n 'metadata' /root/project/model/models/finetune_model.py | head -20")
print(out)

print("\n" + "=" * 60)
print("2. Verify metadata in moe.py")
print("=" * 60)
out = run("grep -n 'metadata' /root/project/model/models/moe.py | head -10")
print(out)

print("\n" + "=" * 60)
print("3. Verify focal_loss alpha=0.25 in finetune_config.yaml")
print("=" * 60)
out = run("grep -A2 'focal_loss' /root/project/model/configs/finetune_config.yaml")
print(out)

print("\n" + "=" * 60)
print("4. Verify overlap_ratio=0.5 in astock_dataset.py")
print("=" * 60)
out = run("grep 'overlap_ratio' /root/project/model/data/astock_dataset.py | head -5")
print(out)

print("\n" + "=" * 60)
print("5. Verify XGBoost in baselines.py")
print("=" * 60)
out = run("grep -i 'xgb\\|xgboost\\|GradientBoosting' /root/project/model/evaluation/baselines.py | head -10")
print(out)

# 2. Check all result JSON files
print("\n" + "=" * 60)
print("6. Phase4 comparison_results.json (key metrics)")
print("=" * 60)
out = run("cat /root/project/outputs/phase4/comparison_results.json")
try:
    data = json.loads(out)
    for name in sorted(data.keys()):
        m = data[name]
        f1 = m.get('f1', 0)
        mcc = m.get('mcc', 0)
        print(f"  {name:<25} F1={f1:.4f}  MCC={mcc:.4f}")
except:
    print(out[:500])

print("\n" + "=" * 60)
print("7. Finetune_v2 comparison_v2.json")
print("=" * 60)
out = run("cat /root/project/outputs/finetune_v2/comparison_v2.json")
try:
    data = json.loads(out)
    for name in sorted(data.keys()):
        m = data[name]
        f1 = m.get('f1', 0)
        mcc = m.get('mcc', 0)
        print(f"  {name:<25} F1={f1:.4f}  MCC={mcc:.4f}")
except:
    print(out[:500])

print("\n" + "=" * 60)
print("8. baselines_v2 log (tail)")
print("=" * 60)
out = run("tail -30 /root/project/baselines_v2_stdout.log 2>/dev/null || echo 'NOT_FOUND'")
print(out)

print("\n" + "=" * 60)
print("9. GPU status")
print("=" * 60)
out = run("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader")
print(out)

print("\n" + "=" * 60)
print("10. Python/PyTorch version")
print("=" * 60)
out = run("/root/miniconda3/bin/python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}\")'")
print(out)

ssh.close()
print("\nVerification complete!")
