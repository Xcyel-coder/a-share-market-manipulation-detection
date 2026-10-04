import os
# -*- coding: utf-8 -*-
import paramiko, sys, time
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)

sftp = ssh.open_sftp()
sftp.put(r"c:\Users\wangm\OneDrive - Audencia\毕业论文\code\run_comprehensive_part2.py",
         "/root/project/run_comprehensive_part2.py")
print("Uploaded run_comprehensive_part2.py")
sftp.close()

ssh.exec_command("pkill -f run_comprehensive || true", timeout=5)
time.sleep(2)

launch_script = """#!/bin/bash
cd /root/project
nohup /root/miniconda3/bin/python run_comprehensive_part2.py > /root/project/outputs/comprehensive/experiment_part2.log 2>&1 &
echo $! > /tmp/part2_pid.txt
echo "PID: $(cat /tmp/part2_pid.txt)"
"""
stdin, stdout, stderr = ssh.exec_command("cat > /tmp/launch_part2.sh << 'SCRIPT'\n" + launch_script + "SCRIPT", timeout=5)
stdout.read()

stdin, stdout, stderr = ssh.exec_command("bash /tmp/launch_part2.sh", timeout=10)
print(stdout.read().decode())
print(stderr.read().decode())

time.sleep(5)

stdin, stdout, stderr = ssh.exec_command("cat /tmp/part2_pid.txt 2>/dev/null", timeout=5)
pid = stdout.read().decode().strip()
print(f"PID: {pid}")

stdin, stdout, stderr = ssh.exec_command(f"ps -p {pid} -o pid,cmd --no-headers 2>/dev/null", timeout=5)
ps_out = stdout.read().decode().strip()
print(f"Process: {ps_out if ps_out else 'NOT FOUND'}")

stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null", timeout=5)
print(f"GPU: {stdout.read().decode().strip()}")

stdin, stdout, stderr = ssh.exec_command("tail -10 /root/project/outputs/comprehensive/experiment_part2.log 2>/dev/null", timeout=10)
print(f"Log:\n{stdout.read().decode()}")

ssh.close()
print("Done.")
