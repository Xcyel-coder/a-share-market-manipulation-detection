"""
Remote GPU server helper — thin wrapper around paramiko.

Usage:
    python remote.py exec "nvidia-smi"
    python remote.py upload local_dir remote_dir
    python remote.py download remote_path local_path
"""
import argparse
import os
import sys
import stat
from pathlib import Path

import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 49397
USER = "root"
PASSWORD = os.environ["AUTODL_SSH_PASSWORD"]

REMOTE_PROJECT_ROOT = "/root/project"


def get_ssh_client() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    return ssh


def remote_exec(cmd: str, timeout: int = 300, print_output: bool = True) -> tuple[int, str, str]:
    """Execute a command on the remote server and return (exit_code, stdout, stderr)."""
    ssh = get_ssh_client()
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        if print_output:
            if out.strip():
                print(out.strip())
            if err.strip():
                print(f"[stderr] {err.strip()}", file=sys.stderr)
        return exit_code, out, err
    finally:
        ssh.close()


def upload_dir(local_dir: str, remote_dir: str):
    """Recursively upload a local directory to the remote server."""
    ssh = get_ssh_client()
    sftp = ssh.open_sftp()
    try:
        local_path = Path(local_dir)
        for local_file in local_path.rglob("*"):
            if local_file.is_file():
                rel = local_file.relative_to(local_path)
                remote_path = f"{remote_dir}/{rel.as_posix()}"
                remote_parent = os.path.dirname(remote_path)
                _sftp_makedirs(sftp, remote_parent)
                print(f"  {rel.as_posix()}")
                sftp.put(str(local_file), remote_path)
        print(f"Upload complete → {remote_dir}")
    finally:
        sftp.close()
        ssh.close()


def upload_file(local_file: str, remote_path: str):
    """Upload a single file to the remote server."""
    ssh = get_ssh_client()
    sftp = ssh.open_sftp()
    try:
        remote_parent = os.path.dirname(remote_path)
        _sftp_makedirs(sftp, remote_parent)
        sftp.put(local_file, remote_path)
        print(f"Uploaded → {remote_path}")
    finally:
        sftp.close()
        ssh.close()


def download_file(remote_path: str, local_path: str):
    """Download a file from the remote server."""
    ssh = get_ssh_client()
    sftp = ssh.open_sftp()
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, local_path)
        print(f"Downloaded → {local_path}")
    finally:
        sftp.close()
        ssh.close()


def _sftp_makedirs(sftp: paramiko.SFTPClient, remote_dir: str):
    """Recursively create remote directories (like mkdir -p)."""
    if remote_dir in ("", "/"):
        return
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        _sftp_makedirs(sftp, os.path.dirname(remote_dir))
        try:
            sftp.mkdir(remote_dir)
        except IOError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Remote GPU server helper")
    sub = parser.add_subparsers(dest="action")

    exec_p = sub.add_parser("exec", help="Execute remote command")
    exec_p.add_argument("cmd", help="Command to execute")
    exec_p.add_argument("--timeout", type=int, default=300)

    up_p = sub.add_parser("upload", help="Upload directory")
    up_p.add_argument("local", help="Local directory")
    up_p.add_argument("remote", help="Remote directory")

    down_p = sub.add_parser("download", help="Download file")
    down_p.add_argument("remote", help="Remote path")
    down_p.add_argument("local", help="Local path")

    args = parser.parse_args()

    if args.action == "exec":
        code, _, _ = remote_exec(args.cmd, timeout=args.timeout)
        sys.exit(code)
    elif args.action == "upload":
        upload_dir(args.local, args.remote)
    elif args.action == "download":
        download_file(args.remote, args.local)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
