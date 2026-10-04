import os
import paramiko, json, os
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.westb.seetacloud.com", port=49397, username="root", password=os.environ["AUTODL_SSH_PASSWORD"], timeout=15)
sftp = ssh.open_sftp()
local = os.path.join(os.path.dirname(__file__), "model", "outputs", "phase4", "backtest_results.json")
sftp.get("/root/project/outputs/phase4/backtest_results.json", local)
sftp.close()
ssh.close()
with open(local, encoding="utf-8") as f:
    d = json.load(f)
print(f"Cases warned: {d['cases_warned']}/{d['total_cases']}")
print(f"Warning rate: {d['warning_rate']:.0%}")
print(f"Avg loss if held: {d['avg_loss_if_held_pct']:.2f}%")
print(f"Avg avoided loss: {d['avg_avoided_loss_pct']:.2f}%")
print(f"Avg crash after manip: {d['avg_crash_after_manip_pct']:.2f}%")
print(f"Avg lead days: {d['avg_lead_days']:.1f}")
for r in sorted(d["details"], key=lambda x: x.get("loss_if_held_pct", 0))[:5]:
    print(f"  {r['doc_id']} ({r['stock_code']}): held_loss={r['loss_if_held_pct']:.1f}%, crash={r['crash_pct']:.1f}%, lead={r['lead_days']}d")
