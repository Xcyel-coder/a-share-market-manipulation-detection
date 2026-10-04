import os
import paramiko, json, os
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.westb.seetacloud.com", port=49397, username="root", password=os.environ["AUTODL_SSH_PASSWORD"], timeout=15)
sftp = ssh.open_sftp()
local_dir = os.path.join(os.path.dirname(__file__), "model", "outputs", "finetune_v2")
os.makedirs(local_dir, exist_ok=True)
sftp.get("/root/project/outputs/finetune_v2/comparison_v2.json", os.path.join(local_dir, "comparison_v2.json"))
for lv in ["L0", "L1", "L2", "L3"]:
    d = os.path.join(local_dir, lv)
    os.makedirs(d, exist_ok=True)
    sftp.get(f"/root/project/outputs/finetune_v2/{lv}/summary.json", os.path.join(d, "summary.json"))
sftp.get("/root/project/outputs/rerun.log", os.path.join(local_dir, "rerun.log"))
sftp.close()
ssh.close()
with open(os.path.join(local_dir, "comparison_v2.json")) as f:
    d = json.load(f)
for name in d:
    m = d[name]
    print(f"  {name}: F1={m['f1']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")
print("Downloaded.")
