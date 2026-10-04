import os
# -*- coding: utf-8 -*-
import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westd.seetacloud.com', port=28784, username='root', password=os.environ["AUTODL_SSH_PASSWORD"], timeout=30)
stdin, stdout, stderr = ssh.exec_command("tail -25 /root/project/pretrain_daily_stdout.log 2>/dev/null", timeout=10)
print(stdout.read().decode())
stdin, stdout, stderr = ssh.exec_command("ls -la /root/project/outputs/pretrain_daily/*/best.pt 2>/dev/null", timeout=10)
print(stdout.read().decode())
ssh.close()
