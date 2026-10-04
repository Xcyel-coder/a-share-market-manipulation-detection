import os
# -*- coding: utf-8 -*-
import paramiko, sys, json
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# Upload a proper extraction script
extract_code = '''#!/usr/bin/env python3
import json, torch, numpy as np
from pathlib import Path

base = Path("/root/project/outputs/comprehensive")
output = {}

for level in ["MoE_L0", "MoE_L1", "MoE_L2", "MoE_L3"]:
    level_dir = base / level
    if not level_dir.exists():
        continue
    fold_data = []
    for fold_idx in range(5):
        ckpt_path = level_dir / f"fold_{fold_idx}" / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            m = ckpt.get("metrics", {})
            fold_data.append({"fold": fold_idx, "metrics": m})
    
    key = level.replace("_", "-")
    if fold_data:
        output[key] = {"fold_data": fold_data}
        for mn in ["mcc", "f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc", "specificity"]:
            vals = [fd["metrics"].get(mn, 0) for fd in fold_data if mn in fd["metrics"]]
            if vals:
                output[key][mn] = round(float(np.mean(vals)), 4)
                output[key][f"{mn}_std"] = round(float(np.std(vals)), 4)

print(json.dumps(output, indent=2, default=str))
'''

sftp = ssh.open_sftp()
with sftp.open("/tmp/extract_metrics.py", "w") as f:
    f.write(extract_code)
sftp.close()

stdin, stdout, stderr = ssh.exec_command(
    "/root/miniconda3/bin/python /tmp/extract_metrics.py", timeout=30)
out = stdout.read().decode()
err = stderr.read().decode()
if err and ("Error" in err or "Traceback" in err):
    print(f"ERROR: {err}")
else:
    print(out)

ssh.close()
