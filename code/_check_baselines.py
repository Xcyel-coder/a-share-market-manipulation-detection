import os
import paramiko, json
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.westb.seetacloud.com", port=49397, username="root", password=os.environ["AUTODL_SSH_PASSWORD"], timeout=15)

_, out, _ = ssh.exec_command("tail -25 /root/project/baselines_v2_stdout.log", timeout=10)
print(out.read().decode())

_, out2, _ = ssh.exec_command("ps aux | grep run_baselines | grep -v grep | wc -l", timeout=5)
print(f"Running: {out2.read().decode().strip()}")

_, out3, _ = ssh.exec_command("cat /root/project/outputs/baselines_v2/baselines_v2.json 2>/dev/null", timeout=5)
data = out3.read().decode().strip()
if data:
    d = json.loads(data)
    print("\n=== BASELINES V2 RESULTS ===")
    for name in d:
        m = d[name]
        print(f"  {name}: F1={m['f1']:.4f} MCC={m['mcc']:.4f} AUC={m['roc_auc']:.4f}")
else:
    print("baselines_v2.json not yet available")

ssh.close()
