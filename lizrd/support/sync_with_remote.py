from contextlib import contextmanager
import copy
import getpass
import os
import subprocess
from typing import Generator
from fabric import Connection
from argparse import ArgumentParser
from git import Repo
import paramiko.ssh_exception
from lizrd.support.code_versioning_support import find_git_root

_HOSTS_TO_PASSPHRASES = {}


@contextmanager
def ConnectWithPassphrase(*args, **kwargs) -> Generator[Connection, None, None]:
    """Connect to a remote host using a passphrase if the key is encrypted. The passphrase is preserved for subsequent connections to the same host."""
    try:
        connection = Connection(*args, **kwargs)
        connection.run('echo "Connection successful."')
        yield connection
    except paramiko.ssh_exception.PasswordRequiredException as e:
        if connection.host not in _HOSTS_TO_PASSPHRASES:
            passphrase = getpass.getpass(
                f"SSH key encrypted, provide the passphrase ({connection.host}): "
            )
            _HOSTS_TO_PASSPHRASES[connection.host] = passphrase
        else:
            passphrase = _HOSTS_TO_PASSPHRASES[connection.host]
        kwargs["connect_kwargs"] = copy.deepcopy(
            kwargs.get("connect_kwargs", {})
        )  # avoid modifying the original connect_kwargs
        kwargs["connect_kwargs"]["passphrase"] = passphrase
        connection = Connection(*args, **kwargs)
        yield connection
    finally:
        connection.close()


def cd_to_root_dir():
    git_root = find_git_root()
    repo = Repo(git_root)
    assert repo.remotes.origin.url in [
        "git@github.com:llm-random/llm-random.git",
    ], "You're not in the right repo! Move to the llm-random folder, and make sure your origin is the llm-random repo. Aborting..."
    os.chdir(git_root)


def rsync_to_remote(host, local_dir):
    try:
        with ConnectWithPassphrase(host) as connection:
            base_dir = get_base_directory(connection)
            proxy_command = get_proxy_command(connection)
        rsync_command = [
            "rsync",
            "--compress",
            "--recursive",
            "--links",
            "--perms",
            "--human-readable",
            "--stats",
            f"--rsh={proxy_command}",
            "--exclude=*.pyc",
            local_dir,
            f"{host}:{base_dir}",
        ]
        print(f"Syncing {local_dir} to {host}:{base_dir}...")
        subprocess.run(rsync_command)
        print("Sync complete.")
        return base_dir
    except Exception as e:
        raise Exception(f"[RSYNC ERROR]: An error occurred during rsync: {str(e)}")


def get_base_directory(connection):
    if connection.host == "athena.cyfronet.pl":
        base_dir = (
            f"/net/pr2/projects/plgrid/plggllmeffi/{connection.user[3:]}/llm-random"
        )
    else:
        base_dir = f"~/llm-random"
    return base_dir


def get_proxy_command(connection):
    if connection.host == "4124gs01":
        with ConnectWithPassphrase(connection.ssh_config["proxyjump"]) as cc:
            proxy_command = f"'ssh -A -J {cc.user}@{cc.host}'"
    else:
        proxy_command = "ssh"
    return proxy_command


def run_remote_script(host, script):
    try:
        with ConnectWithPassphrase(host) as c:
            result = c.run(script)
    except Exception as e:
        raise Exception(f"An error occurred while running the script: {str(e)}")


def set_up_permissions(host):
    try:
        with ConnectWithPassphrase(host) as connection:
            path = f"{get_base_directory(connection)}/lizrd/scripts/grid_entrypoint_athena.sh"
            print(f"Changing permissions for {path}...")
            connection.run(f"chmod +x {path}")
            print("The permissions for the script have been changed successfully.")
    except Exception as e:
        raise Exception(
            f"The permissions change for the script failed. Error: {str(e)}"
        )


if __name__ == "__main__":
    # set up argparse for hostname, config
    parser = ArgumentParser()
    parser.add_argument("--host", type=str)
    # create parser
    args = parser.parse_args()

    cd_to_root_dir()
    root_dir = os.getcwd()
    base_dir = rsync_to_remote(args.host, root_dir + "/lizrd")
    _ = rsync_to_remote(args.host, root_dir + "/research")
    _ = rsync_to_remote(args.host, root_dir + "/.versioningignore")
    set_up_permissions(args.host)
    # WRITE root_dir to temp file for run_exp_remotely.sh
    with open("base_dir.txt", "w") as f:
        f.write(base_dir)
