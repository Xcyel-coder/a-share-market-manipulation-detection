"""Fix L3 device issue and re-run only L2+L3."""
import os
import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
sftp = ssh.open_sftp()

# Read the remote script
with sftp.file("/root/project/run_rerun.py", "r") as f:
    code = f.read().decode()

# Fix 1: Add model.to(device) before get_expert_activations
old = "        if level_name == \"L3\" and fold_idx == 0:\n            act_loader"
new = "        if level_name == \"L3\" and fold_idx == 0:\n            model.to(device)\n            act_loader"
code = code.replace(old, new)

# Fix 2: Only run L2 and L3 (L0 and L1 already done)
code = code.replace(
    'for level_name in ["L0", "L1", "L2", "L3"]:',
    'for level_name in ["L2", "L3"]:'
)

# Fix 3: Load L0/L1 results from existing files
load_existing = '''
# Load existing L0/L1 results
import json as _json
for _lv in ["L0", "L1"]:
    _p = OUTPUT_DIR / _lv / "summary.json"
    if _p.exists():
        with open(_p) as _f:
            all_results[f"MoE-{_lv}"] = _json.load(_f)
        logger.info(f"Loaded existing {_lv} results")

'''
code = code.replace("all_results = {}\n", "all_results = {}\n" + load_existing)

with sftp.file("/root/project/run_rerun.py", "w") as f:
    f.write(code)

sftp.close()

# Launch
ssh.exec_command("pkill -f run_rerun 2>/dev/null", timeout=5)
import time; time.sleep(2)

cmd = "cd /root/project && nohup /root/miniconda3/bin/python -u run_rerun.py > rerun_stdout.log 2>&1 &"
ssh.exec_command(cmd, timeout=5)
time.sleep(3)

_, out, _ = ssh.exec_command("ps aux | grep run_rerun | grep -v grep", timeout=5)
ps = out.read().decode()
print("Running!" if "run_rerun" in ps else "NOT running")
print(ps.strip())

ssh.close()
print("Done")
