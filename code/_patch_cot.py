"""Patch the remote script to fix CoT loading, then relaunch."""
import os
import paramiko
import time

HOST = "connect.westd.seetacloud.com"
PORT = 28784
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]
PYTHON = "/root/miniconda3/bin/python"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
sftp = ssh.open_sftp()


def run(cmd, timeout=30):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    if out:
        print(out[:600])
    return out


# 1. Kill
print("=== Kill old process ===")
run("pkill -f 'run_clean_rerun' 2>/dev/null; sleep 2; echo 'killed'")

# 2. Read, patch, write back
print("\n=== Patching script ===")
with sftp.file("/root/project/run_clean_rerun.py", "r") as f:
    content = f.read().decode("utf-8")

old_cot = 'cot_embeddings = {}\nif Path(COT_PATH).exists():\n    with open(COT_PATH) as f:\n        cot_data = json.load(f)\n    for entry in cot_data:\n        doc_id = entry.get("doc_id", "")\n        emb = entry.get("embedding")\n        if doc_id and emb:\n            cot_embeddings[doc_id] = np.array(emb, dtype=np.float32)\nlogger.info(f"CoT embeddings: {len(cot_embeddings)}")'

new_cot = '''cot_embeddings = {}
COT_NPZ = "/root/project/model/outputs/cot/cot_embeddings.npz"
if Path(COT_NPZ).exists():
    npz_data = np.load(COT_NPZ)
    cot_embeddings = {k: npz_data[k] for k in npz_data.files}
    logger.info(f"CoT embeddings loaded from NPZ: {len(cot_embeddings)}")
else:
    logger.warning("No CoT embeddings NPZ found!")'''

if old_cot in content:
    content = content.replace(old_cot, new_cot)
    with sftp.file("/root/project/run_clean_rerun.py", "w") as f:
        f.write(content)
    print("Patch applied successfully!")
else:
    print("WARNING: Could not find exact match. Trying line-by-line...")
    lines = content.split("\n")
    new_lines = []
    skip = False
    patched = False
    for i, line in enumerate(lines):
        if 'cot_embeddings = {}' in line and not patched:
            new_lines.append(new_cot)
            skip = True
            patched = True
            continue
        if skip:
            if 'logger.info(f"CoT embeddings:' in line:
                skip = False
                continue
            continue
        new_lines.append(line)
    
    if patched:
        with sftp.file("/root/project/run_clean_rerun.py", "w") as f:
            f.write("\n".join(new_lines))
        print("Patch applied (line-by-line)!")
    else:
        print("FAILED to patch!")

# 3. Verify
print("\n=== Verify patch ===")
run("grep -n 'cot_embeddings\\|COT_NPZ' /root/project/run_clean_rerun.py | head -10")

# 4. Clean and relaunch
print("\n=== Relaunch ===")
run("rm -rf /root/project/outputs/clean_rerun/*; mkdir -p /root/project/outputs/clean_rerun")
cmd = f"cd /root/project && nohup {PYTHON} -u run_clean_rerun.py > clean_rerun_stdout.log 2>&1 &"
ssh.exec_command(cmd, timeout=5)
time.sleep(8)

run("ps aux | grep run_clean_rerun | grep -v grep")

print("\n=== Initial log ===")
run("head -45 /root/project/clean_rerun_stdout.log")

sftp.close()
ssh.close()
print("\nDone!")
