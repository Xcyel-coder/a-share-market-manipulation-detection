"""Check new AutoDL instance for data from old instance."""
import os
import paramiko

HOST = "connect.westd.seetacloud.com"
PORT = 28784
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)


def run(cmd):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err and "No such" not in err:
        print(f"  [stderr] {err[:300]}")
    return out


print("=" * 60)
print("1. /root/project structure")
print("=" * 60)
run("ls -la /root/project/ 2>/dev/null || echo '/root/project NOT FOUND'")

print("\n" + "=" * 60)
print("2. Pretrain weights")
print("=" * 60)
run("find /root/project/outputs/pretrain -name '*.pt' -exec ls -lh {} \\; 2>/dev/null || echo 'NOT FOUND'")

print("\n" + "=" * 60)
print("3. Finetune checkpoints (v1)")
print("=" * 60)
run("find /root/project/outputs/finetune -name 'best.pt' -exec ls -lh {} \\; 2>/dev/null || echo 'NOT FOUND'")

print("\n" + "=" * 60)
print("4. Finetune_v2 results")
print("=" * 60)
run("ls -la /root/project/outputs/finetune_v2/ 2>/dev/null || echo 'NOT FOUND'")

print("\n" + "=" * 60)
print("5. Phase4 results")
print("=" * 60)
run("ls -la /root/project/outputs/phase4/ 2>/dev/null || echo 'NOT FOUND'")

print("\n" + "=" * 60)
print("6. Crypto parquet files")
print("=" * 60)
run("find /root/project/data/raw/crypto -name '*.parquet' 2>/dev/null | sort")
run("du -sh /root/project/data/raw/crypto/ 2>/dev/null || echo 'NOT FOUND'")

print("\n" + "=" * 60)
print("7. Astock parquet files")
print("=" * 60)
run("find /root/project/data/raw/astock -name '*.parquet' 2>/dev/null | sort")

print("\n" + "=" * 60)
print("8. Extracted cases")
print("=" * 60)
out = run("ls /root/project/data/processed/extracted_cases/ 2>/dev/null | wc -l")
print(f"  Case files: {out}")

print("\n" + "=" * 60)
print("9. CoT results")
print("=" * 60)
run("find /root/project -name 'cot_results.json' -exec ls -lh {} \\; 2>/dev/null || echo 'NOT FOUND'")
run("find /root/project -name 'cot_embeddings.npz' -exec ls -lh {} \\; 2>/dev/null || echo 'NOT FOUND'")

print("\n" + "=" * 60)
print("10. Model code")
print("=" * 60)
run("ls /root/project/model/models/ 2>/dev/null || echo 'NOT FOUND'")
run("ls /root/project/model/data/ 2>/dev/null || echo 'NOT FOUND'")

print("\n" + "=" * 60)
print("11. Disk usage summary")
print("=" * 60)
run("du -sh /root/project/ 2>/dev/null || echo 'NOT FOUND'")
run("du -sh /root/project/outputs/ 2>/dev/null || echo 'NOT FOUND'")
run("du -sh /root/project/data/ 2>/dev/null || echo 'NOT FOUND'")
run("df -h /root")

ssh.close()
print("\nDone!")
