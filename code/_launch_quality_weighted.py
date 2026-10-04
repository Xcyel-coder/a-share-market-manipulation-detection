import os
# -*- coding: utf-8 -*-
"""Upload verified CSV + experiment script to server and launch."""
import sys, time, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUNBUFFERED"] = "1"

import paramiko

HOST = "connect.westd.seetacloud.com"
PORT = 28784
USER = "root"
PASS = os.environ["AUTODL_SSH_PASSWORD"]

BASE = r"c:\Users\wangm\OneDrive - Audencia\毕业论文"

FILES_TO_UPLOAD = [
    (os.path.join(BASE, "manipulation_cases_verified.csv"),
     "/root/project/data/manipulation_cases_verified.csv"),
    (os.path.join(BASE, "code", "run_quality_weighted_experiment.py"),
     "/root/project/run_quality_weighted_experiment.py"),
]

P = lambda *a, **kw: print(*a, **kw, flush=True)


def main():
    P("连接服务器...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = ssh.open_sftp()
    P("已连接")

    # Check server state
    P("\n--- 服务器状态 ---")
    for cmd in [
        "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader",
        "ls /root/project/data/processed/extracted_cases/ | head -5",
        "ls /root/project/data/raw/astock/ | head -5",
        "ls /root/project/outputs/pretrain_daily/align_daily/best.pt 2>/dev/null && echo 'Pretrain checkpoint OK' || echo 'NO pretrain checkpoint'",
    ]:
        _, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode().strip()
        if out:
            P(f"  {out}")

    # Upload files
    P("\n--- 上传文件 ---")
    for local, remote in FILES_TO_UPLOAD:
        P(f"  {os.path.basename(local)} -> {remote}")
        sftp.put(local, remote)
    P("上传完成")

    # Verify CSV
    P("\n--- 验证 CSV ---")
    _, stdout, _ = ssh.exec_command("wc -l /root/project/data/manipulation_cases_verified.csv")
    P(f"  CSV 行数: {stdout.read().decode().strip()}")
    _, stdout, _ = ssh.exec_command("head -1 /root/project/data/manipulation_cases_verified.csv")
    P(f"  表头: {stdout.read().decode().strip()[:120]}...")

    # Launch experiment
    P("\n--- 启动实验 ---")
    launch_cmd = (
        "cd /root/project && "
        "nohup /root/miniconda3/bin/python3 -u run_quality_weighted_experiment.py "
        "> /root/project/outputs/quality_weighted/run.log 2>&1 &"
    )
    ssh.exec_command(f"mkdir -p /root/project/outputs/quality_weighted")
    ssh.exec_command(launch_cmd)
    time.sleep(3)

    # Verify launch
    _, stdout, _ = ssh.exec_command("ps aux | grep quality_weighted | grep -v grep")
    ps = stdout.read().decode().strip()
    if ps:
        P(f"  进程已启动: {ps[:120]}")
    else:
        P("  警告: 进程可能未成功启动")
        _, stdout, _ = ssh.exec_command("cat /root/project/outputs/quality_weighted/run.log 2>/dev/null | tail -20")
        P(stdout.read().decode().strip())

    # Quick check log
    time.sleep(5)
    _, stdout, _ = ssh.exec_command("tail -30 /root/project/outputs/quality_weighted/run.log 2>/dev/null")
    log = stdout.read().decode().strip()
    if log:
        P(f"\n--- 初始日志 ---")
        P(log)

    sftp.close()
    ssh.close()
    P("\n完成")


if __name__ == "__main__":
    main()
