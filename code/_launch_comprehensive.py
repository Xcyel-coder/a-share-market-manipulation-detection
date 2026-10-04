import os
# -*- coding: utf-8 -*-
import paramiko, sys, time
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

# 1. Upload script
sftp = ssh.open_sftp()
sftp.put(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\run_comprehensive_experiment.py",
         "/root/project/run_comprehensive_experiment.py")
print("Uploaded run_comprehensive_experiment.py")
sftp.close()

# 2. Kill any existing training
ssh.exec_command("pkill -f run_comprehensive_experiment || true", timeout=5)
ssh.exec_command("pkill -f run_ablation_study || true", timeout=5)
ssh.exec_command("pkill -f run_expanded_finetune || true", timeout=5)
time.sleep(2)

# 3. Create output dir
ssh.exec_command("mkdir -p /root/project/outputs/comprehensive", timeout=5)

# 4. Create launch script
launch_script = """#!/bin/bash
cd /root/project
nohup /root/miniconda3/bin/python run_comprehensive_experiment.py > /root/project/outputs/comprehensive/experiment.log 2>&1 &
echo $! > /tmp/comprehensive_pid.txt
echo "PID: $(cat /tmp/comprehensive_pid.txt)"
"""
stdin, stdout, stderr = ssh.exec_command("cat > /tmp/launch_comprehensive.sh << 'SCRIPT'\n" + launch_script + "SCRIPT", timeout=5)
stdout.read()

# 5. Launch
stdin, stdout, stderr = ssh.exec_command("bash /tmp/launch_comprehensive.sh", timeout=10)
print(stdout.read().decode())
print(stderr.read().decode())

time.sleep(3)

# 6. Verify
stdin, stdout, stderr = ssh.exec_command("cat /tmp/comprehensive_pid.txt 2>/dev/null", timeout=5)
pid = stdout.read().decode().strip()
print(f"Process PID: {pid}")

stdin, stdout, stderr = ssh.exec_command(f"ps -p {pid} -o pid,cmd --no-headers 2>/dev/null", timeout=5)
ps_out = stdout.read().decode().strip()
print(f"Process status: {ps_out if ps_out else 'NOT FOUND'}")

stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null", timeout=5)
print(f"GPU: {stdout.read().decode().strip()}")

stdin, stdout, stderr = ssh.exec_command("tail -5 /root/project/outputs/comprehensive/experiment.log 2>/dev/null", timeout=5)
print(f"Log tail:\n{stdout.read().decode()}")

ssh.close()
print("Done.")
