"""
Quick way to setup the AIxCC nginx Challenge Project in a Path of your choice
"""

import argparse
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from typing import List
from typing import Union

SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(command: Union[str, List[str]], cwd: Path = Path.cwd(), shell: bool = False) -> CompletedProcess[bytes]:
    """Run a shell command and print it beforehand"""
    print(f"Running command: {' '.join(command) if isinstance(command, list) else command}")
    result = subprocess.run(command, shell=shell, cwd=cwd, check=True)
    return result


def check_yq() -> None:
    """checks yq against required version"""
    try:
        result = subprocess.run(
            ["yq", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"yq version check failed: {e}")
        sys.exit(1)
    expected = "yq (https://github.com/mikefarah/yq/) version v4.44.1"
    actual = result.stdout.strip()
    if actual != expected:
        print("yq version mismatch")
        print(f"Expected: {expected}")
        print(f"Found:    {actual}")
        print("Please update your venv:")
        print("curl -L https://github.com/mikefarah/yq/releases/download/v4.44.1/yq_linux_amd64 -o venv/bin/yq")
        print("chmod +x venv/bin/yq")
        sys.exit(1)


def docker_running() -> bool:
    """checks if docker is running"""
    try:
        run_command(["docker", "info"])
        return True
    except subprocess.CalledProcessError:
        return False


def clone_and_prep(output_dir: Path, build_project: bool = True) -> None:
    """
    Clone the AIxCC nginx Challenge Project, prepare and build it and copy the compile_commands.json into /work
    -> get ready-to-use nginx CP in output_dir
    """
    nginx_dir = output_dir.resolve() / "challenge-004-nginx-cp"

    # get Nginx CP from DARAPA repo
    run_command(["git", "clone", "https://github.com/aixcc-public/challenge-004-nginx-cp.git", str(nginx_dir)])

    # Prepare inside cloned repo
    run_command(["make", "cpsrc-prepare"], cwd=nginx_dir)
    run_command(["make", "docker-pull"], cwd=nginx_dir)
    if build_project:
        run_command(["./run.sh", "build"], cwd=nginx_dir)

    # Copy compile_commands.json
    compile_commands_src = SCRIPT_DIR / "compile_commands.json"
    compile_commands_dst = nginx_dir / "work"

    if not compile_commands_src.exists():
        print(f"Error: {compile_commands_src} does not exist.", file=sys.stderr)
        sys.exit(1)
    run_command(["cp", str(compile_commands_src), str(compile_commands_dst)])


def main() -> None:
    """
    Clone nginx CP to output dir and start a 20 sample test series on it using Qwen Coder 2.5
    Output will be printed as table and may be added to README for future reference
    """
    parser = argparse.ArgumentParser(description="Automate nginx-cp challenge setup.")
    parser.add_argument("-o", type=Path, default=".", help="Directory for output (default; currenct directory).")
    args = parser.parse_args()

    # get absolute Paths
    output_dir: Path = args.o.resolve()

    # check for docker
    docker_active = docker_running()
    if docker_active is False:
        print("Please start the docker daemon, then try again.")
        sys.exit(1)

    # check for yq
    check_yq()

    # prepare challenge project
    clone_and_prep(output_dir)


if __name__ == "__main__":
    main()
