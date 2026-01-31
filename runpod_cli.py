#!/usr/bin/env python3
"""
CLI tool for managing modded-nanogpt training on RunPod.

Usage:
    python runpod_cli.py ssh-keygen
    python runpod_cli.py configure <connection-string>
    python runpod_cli.py sync-code
    python runpod_cli.py fetch-data [--shards N]
    python runpod_cli.py setup
    python runpod_cli.py train start [--single-gpu]
    python runpod_cli.py train check [--lines N]
    python runpod_cli.py train stop
    python runpod_cli.py gpu-status
"""

import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import click
except ImportError:
    print("click is required. Install with: pip install click")
    sys.exit(1)

# Paths
SSH_DIR = Path.home() / ".ssh"
SSH_KEY_PATH = SSH_DIR / "id_ed25519_runpod"
SSH_CONFIG_PATH = SSH_DIR / "config"
SSH_SOCKETS_DIR = SSH_DIR / "sockets"
REPO_DIR = Path(__file__).parent.resolve()

# Remote paths
REMOTE_WORKSPACE = "/workspace"
REMOTE_REPO = f"{REMOTE_WORKSPACE}/modded-nanogpt"
REMOTE_PIDFILE = f"{REMOTE_REPO}/train.pid"
REMOTE_LOGFILE = f"{REMOTE_REPO}/train.log"

SSH_CONFIG_TEMPLATE = """
# RunPod instance - added by runpod_cli.py
Host runpod
    HostName {hostname}
    Port {port}
    User root
    IdentityFile {keyfile}
    ControlMaster auto
    ControlPath {sockets_dir}/%r@%h-%p
    ControlPersist 600
    ServerAliveInterval 60
    ServerAliveCountMax 3
    StrictHostKeyChecking accept-new
"""


def run_cmd(cmd: list[str], check: bool = True, capture: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and optionally capture output."""
    if capture:
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
        kwargs.setdefault("text", True)
    return subprocess.run(cmd, check=check, **kwargs)


def ssh_cmd(remote_cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command on the remote host via SSH."""
    return run_cmd(["ssh", "runpod", remote_cmd], check=check, capture=capture)


def ssh_available() -> bool:
    """Check if SSH to runpod is configured and working."""
    result = run_cmd(["ssh", "-o", "ConnectTimeout=5", "runpod", "echo ok"], check=False, capture=True)
    return result.returncode == 0


@click.group()
def cli():
    """CLI tool for managing modded-nanogpt training on RunPod."""
    pass


@cli.command("ssh-keygen")
def cmd_ssh_keygen():
    """Generate an SSH key for RunPod and show setup instructions."""
    SSH_DIR.mkdir(mode=0o700, exist_ok=True)
    SSH_SOCKETS_DIR.mkdir(mode=0o700, exist_ok=True)

    if SSH_KEY_PATH.exists():
        click.echo(f"SSH key already exists at {SSH_KEY_PATH}")
        click.echo(f"Public key: {SSH_KEY_PATH}.pub")
    else:
        click.echo(f"Generating SSH key at {SSH_KEY_PATH}...")
        run_cmd([
            "ssh-keygen", "-t", "ed25519",
            "-f", str(SSH_KEY_PATH),
            "-N", "",  # No passphrase
            "-C", "runpod-modded-nanogpt"
        ])
        click.echo("SSH key generated.")

    # Show the public key
    pub_key = (SSH_KEY_PATH.with_suffix(".pub")).read_text().strip()
    click.echo()
    click.echo("=" * 60)
    click.echo("PUBLIC KEY - Give this to the user to add to RunPod:")
    click.echo("=" * 60)
    click.echo(pub_key)
    click.echo("=" * 60)
    click.echo()
    click.echo("Next steps:")
    click.echo("1. Provide the public key above to the user")
    click.echo("2. User adds it to RunPod Settings > SSH Public Keys")
    click.echo("3. User starts an H100 instance and provides the connection string")
    click.echo("4. Run: python runpod_cli.py configure '<connection-string>'")


@cli.command("configure")
@click.argument("connection_string")
def cmd_configure(connection_string: str):
    """Configure SSH for RunPod using the connection string from the RunPod UI.

    Example: python runpod_cli.py configure 'ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519'
    """
    # Parse connection string: ssh root@<host> -p <port> -i <keyfile>
    # or just: root@<host> -p <port>
    match = re.search(r"(?:ssh\s+)?(?:\w+@)?([\d.]+)\s+-p\s+(\d+)", connection_string)
    if not match:
        click.echo("Error: Could not parse connection string.", err=True)
        click.echo("Expected format: ssh root@<ip> -p <port> -i ~/.ssh/...", err=True)
        sys.exit(1)

    hostname = match.group(1)
    port = match.group(2)

    # Ensure key exists
    if not SSH_KEY_PATH.exists():
        click.echo(f"SSH key not found at {SSH_KEY_PATH}", err=True)
        click.echo("Run 'python runpod_cli.py ssh-keygen' first", err=True)
        sys.exit(1)

    # Ensure sockets directory exists
    SSH_SOCKETS_DIR.mkdir(mode=0o700, exist_ok=True)

    # Read existing config
    if SSH_CONFIG_PATH.exists():
        config_content = SSH_CONFIG_PATH.read_text()
    else:
        config_content = ""

    # Remove old runpod config if present
    config_lines = config_content.split("\n")
    new_lines = []
    skip_until_next_host = False
    for line in config_lines:
        if line.strip().startswith("Host runpod"):
            skip_until_next_host = True
            continue
        if skip_until_next_host:
            if line.strip().startswith("Host ") or (line.strip() and not line[0].isspace()):
                skip_until_next_host = False
            else:
                continue
        new_lines.append(line)

    # Remove trailing "# RunPod instance" comment lines
    while new_lines and new_lines[-1].strip().startswith("# RunPod instance"):
        new_lines.pop()

    # Add new config
    new_config = SSH_CONFIG_TEMPLATE.format(
        hostname=hostname,
        port=port,
        keyfile=SSH_KEY_PATH,
        sockets_dir=SSH_SOCKETS_DIR
    )

    final_config = "\n".join(new_lines).rstrip() + "\n" + new_config

    SSH_CONFIG_PATH.write_text(final_config)
    SSH_CONFIG_PATH.chmod(0o600)

    click.echo(f"SSH config updated: {SSH_CONFIG_PATH}")
    click.echo(f"  Host: {hostname}")
    click.echo(f"  Port: {port}")
    click.echo()

    # Test connection
    click.echo("Testing connection...")
    if ssh_available():
        click.echo("Connection successful!")
        click.echo()
        click.echo("Next: Run 'python runpod_cli.py setup' to set up the remote environment")
    else:
        click.echo("Connection failed. Check that:", err=True)
        click.echo("  1. The RunPod instance is running", err=True)
        click.echo("  2. Your public key is added to RunPod", err=True)
        click.echo("  3. The connection string is correct", err=True)
        sys.exit(1)


@cli.command("sync-code")
@click.option("--dry-run", is_flag=True, help="Show what would be synced without syncing")
def cmd_sync_code(dry_run: bool):
    """Sync the local code to the remote RunPod instance."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    # Ensure remote directory exists
    ssh_cmd(f"mkdir -p {REMOTE_REPO}")

    rsync_args = [
        "rsync", "-rlptvz", "--progress",  # -a without -o/-g to avoid chown errors
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".pytest_cache",
        "--exclude", "logs/",
        "--exclude", "data/fineweb10B/",
        "--exclude", "train.log",
        "--exclude", "train.pid",
    ]

    if dry_run:
        rsync_args.append("--dry-run")

    rsync_args.extend([
        "-e", "ssh",
        f"{REPO_DIR}/",
        f"runpod:{REMOTE_REPO}/"
    ])

    click.echo(f"Syncing {REPO_DIR} to runpod:{REMOTE_REPO}...")
    run_cmd(rsync_args)
    click.echo("Sync complete.")


@cli.command("fetch-data")
@click.option("--shards", default=None, type=int, help="Number of shards to download (default: all 103)")
def cmd_fetch_data(shards: int | None):
    """Download training data on the remote instance."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    click.echo("Downloading training data on remote instance...")

    if shards:
        cmd = f"python {REMOTE_REPO}/data/cached_fineweb10B.py {shards}"
    else:
        cmd = f"python {REMOTE_REPO}/data/cached_fineweb10B.py"

    # Run with output streaming
    run_cmd(["ssh", "runpod", cmd])
    click.echo("Data download complete.")


@cli.command("setup")
def cmd_setup():
    """Set up the remote environment (install dependencies, sync code)."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    click.echo("Setting up remote environment...")

    # Install tmux and rsync
    click.echo("\n[1/4] Installing tmux and rsync...")
    ssh_cmd("apt-get update -qq && apt-get install -y -qq tmux rsync")

    # Sync code
    click.echo("\n[2/4] Syncing code...")
    ctx = click.Context(cmd_sync_code)
    ctx.invoke(cmd_sync_code, dry_run=False)

    # Install Python dependencies
    click.echo("\n[3/4] Installing Python dependencies...")
    ssh_cmd(f"pip install -q -r {REMOTE_REPO}/requirements.txt")

    # Install PyTorch nightly
    click.echo("\n[4/4] Installing PyTorch nightly (this takes a few minutes)...")
    ssh_cmd("pip install -q torch==2.10.0.dev20251210+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126")

    click.echo("\nSetup complete!")
    click.echo()
    click.echo("Next steps:")
    click.echo("  1. Download training data: python runpod_cli.py fetch-data")
    click.echo("  2. Start training: python runpod_cli.py train start --single-gpu")


@cli.group()
def train():
    """Manage training runs."""
    pass


@train.command("start")
@click.option("--single-gpu", is_flag=True, default=True, help="Use single GPU mode (default: True)")
def cmd_train_start(single_gpu: bool):
    """Start a training run on the remote instance."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    # Check if already running
    result = ssh_cmd(f"test -f {REMOTE_PIDFILE} && ps -p $(cat {REMOTE_PIDFILE}) > /dev/null 2>&1 && echo running || echo stopped", capture=True)
    if result.stdout.strip() == "running":
        click.echo("Training is already running. Use 'train check' to see progress or 'train stop' to stop it.")
        sys.exit(1)

    # Build command
    env_export = "export SINGLE_GPU=1 && " if single_gpu else ""
    train_cmd = "torchrun --nproc_per_node=1 train_gpt.py"

    # Start training with nohup, save pid
    remote_cmd = f"""
        cd {REMOTE_REPO} && {env_export}nohup {train_cmd} > {REMOTE_LOGFILE} 2>&1 & echo $! > {REMOTE_PIDFILE}
    """

    click.echo("Starting training...")
    ssh_cmd(remote_cmd)

    # Verify it started
    import time
    time.sleep(2)
    result = ssh_cmd(f"test -f {REMOTE_PIDFILE} && ps -p $(cat {REMOTE_PIDFILE}) > /dev/null 2>&1 && echo running || echo stopped", capture=True)

    if result.stdout.strip() == "running":
        pid = ssh_cmd(f"cat {REMOTE_PIDFILE}", capture=True).stdout.strip()
        click.echo(f"Training started (PID: {pid})")
        click.echo()
        click.echo("Monitor with: python runpod_cli.py train check")
        click.echo("Stop with: python runpod_cli.py train stop")
    else:
        click.echo("Training failed to start. Check logs:", err=True)
        ssh_cmd(f"tail -50 {REMOTE_LOGFILE}")
        sys.exit(1)


@train.command("check")
@click.option("--lines", "-n", default=30, help="Number of log lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (like tail -f)")
def cmd_train_check(lines: int, follow: bool):
    """Check training status and show recent logs."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    # Check if running
    result = ssh_cmd(f"test -f {REMOTE_PIDFILE} && ps -p $(cat {REMOTE_PIDFILE}) > /dev/null 2>&1 && echo running || echo stopped", capture=True)
    status = result.stdout.strip()

    if status == "running":
        pid = ssh_cmd(f"cat {REMOTE_PIDFILE}", capture=True).stdout.strip()
        click.echo(f"Status: RUNNING (PID: {pid})")
    else:
        click.echo("Status: STOPPED")

    click.echo()
    click.echo(f"=== Last {lines} lines of log ===")

    if follow:
        # Use tail -f for following
        try:
            run_cmd(["ssh", "runpod", f"tail -f {REMOTE_LOGFILE}"])
        except KeyboardInterrupt:
            click.echo("\nStopped following logs.")
    else:
        ssh_cmd(f"tail -{lines} {REMOTE_LOGFILE}", check=False)


@train.command("stop")
def cmd_train_stop():
    """Stop the running training."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    # Check if running
    result = ssh_cmd(f"test -f {REMOTE_PIDFILE} && ps -p $(cat {REMOTE_PIDFILE}) > /dev/null 2>&1 && echo running || echo stopped", capture=True)

    if result.stdout.strip() != "running":
        click.echo("Training is not running.")
        return

    pid = ssh_cmd(f"cat {REMOTE_PIDFILE}", capture=True).stdout.strip()
    click.echo(f"Stopping training (PID: {pid})...")

    # Send SIGTERM, wait, then SIGKILL if needed
    ssh_cmd(f"kill {pid} 2>/dev/null || true", check=False)

    import time
    for _ in range(10):
        time.sleep(1)
        result = ssh_cmd(f"ps -p {pid} > /dev/null 2>&1 && echo running || echo stopped", capture=True)
        if result.stdout.strip() == "stopped":
            break
    else:
        click.echo("Process didn't stop, sending SIGKILL...")
        ssh_cmd(f"kill -9 {pid} 2>/dev/null || true", check=False)

    ssh_cmd(f"rm -f {REMOTE_PIDFILE}", check=False)
    click.echo("Training stopped.")


@cli.command("gpu-status")
def cmd_gpu_status():
    """Show GPU utilization on the remote instance."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    ssh_cmd("nvidia-smi")


@cli.command("shell")
def cmd_shell():
    """Open an interactive shell on the remote instance."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    os.execvp("ssh", ["ssh", "-t", "runpod", f"tmux attach -t train 2>/dev/null || tmux new-session -s train"])


# ============================================================================
# Grid Search Commands
# ============================================================================

@cli.group()
def grid():
    """Manage grid search experiments for RoPE hyperparameters."""
    pass


@grid.command("list")
@click.option("--rope-bases", type=str, default=None, help="Comma-separated RoPE base frequencies (default: 512,1024,2048,4096)")
@click.option("--truncations", type=str, default=None, help="Comma-separated truncation splits (default: 0.25,0.5,0.75)")
@click.option("--replicates", type=int, default=3, help="Number of replicates per config (default: 3)")
def cmd_grid_list(rope_bases: str | None, truncations: str | None, replicates: int):
    """List all grid search configurations."""
    # Global options must come before subcommand
    args = ["python", "grid_search.py", "--replicates", str(replicates)]
    if rope_bases:
        args.extend(["--rope-bases"] + rope_bases.split(","))
    if truncations:
        args.extend(["--truncations"] + truncations.split(","))
    args.append("list")
    run_cmd(args, cwd=REPO_DIR)


@grid.command("run")
@click.argument("config_id", type=int)
@click.option("--rope-bases", type=str, default=None, help="Comma-separated RoPE base frequencies")
@click.option("--truncations", type=str, default=None, help="Comma-separated truncation splits")
@click.option("--replicates", type=int, default=3, help="Number of replicates per config")
@click.option("--single-gpu", is_flag=True, default=True, help="Use single GPU mode (default: True)")
def cmd_grid_run(config_id: int, rope_bases: str | None, truncations: str | None, replicates: int, single_gpu: bool):
    """Run a specific grid search configuration on the remote instance."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    # Build grid_search.py command to get config details
    # Global options must come before subcommand
    local_args = ["python", "grid_search.py", "--replicates", str(replicates)]
    if rope_bases:
        local_args.extend(["--rope-bases"] + rope_bases.split(","))
    if truncations:
        local_args.extend(["--truncations"] + truncations.split(","))
    local_args.append("list")

    # Get configuration from local grid_search.py
    import json
    result = run_cmd(local_args, cwd=REPO_DIR, capture=True)
    lines = result.stdout.strip().split("\n")

    # Parse the config from the list output
    config_line = None
    for line in lines:
        if line.startswith(f"{config_id:3d} "):
            config_line = line
            break

    if not config_line:
        click.echo(f"Error: Config ID {config_id} not found", err=True)
        sys.exit(1)

    # Parse config_line: "  0 rope512_trunc0.25_rep0               512      0.25    0"
    parts = config_line.split()
    config_id_str = parts[1]  # e.g., "rope512_trunc0.25_rep0"
    rope_base_val = parts[2]  # e.g., "512"
    truncation_val = parts[3]  # e.g., "0.25"

    click.echo(f"Running grid config {config_id}: {config_id_str}")
    click.echo(f"  RoPE base: {rope_base_val}")
    click.echo(f"  Truncation: {truncation_val}")

    # Check if already running
    result = ssh_cmd(f"test -f {REMOTE_PIDFILE} && ps -p $(cat {REMOTE_PIDFILE}) > /dev/null 2>&1 && echo running || echo stopped", capture=True)
    if result.stdout.strip() == "running":
        click.echo("Training is already running. Use 'train stop' first.")
        sys.exit(1)

    # Build environment exports
    env_exports = f"export ROPE_BASE={rope_base_val} && export TRUNCATION_SPLIT={truncation_val} && export GRID_CONFIG_ID={config_id_str}"
    if single_gpu:
        env_exports += " && export SINGLE_GPU=1"

    train_cmd = "torchrun --nproc_per_node=1 train_gpt_grid.py"

    # Create unique log file for this config
    log_file = f"{REMOTE_REPO}/logs/grid_{config_id_str}.log"

    # Start training with nohup, save pid
    remote_cmd = f"""
        cd {REMOTE_REPO} && mkdir -p logs && {env_exports} && nohup {train_cmd} > {log_file} 2>&1 & echo $! > {REMOTE_PIDFILE}
    """

    click.echo("Starting training...")
    ssh_cmd(remote_cmd)

    # Verify it started
    import time
    time.sleep(2)
    result = ssh_cmd(f"test -f {REMOTE_PIDFILE} && ps -p $(cat {REMOTE_PIDFILE}) > /dev/null 2>&1 && echo running || echo stopped", capture=True)

    if result.stdout.strip() == "running":
        pid = ssh_cmd(f"cat {REMOTE_PIDFILE}", capture=True).stdout.strip()
        click.echo(f"Training started (PID: {pid})")
        click.echo(f"Log file: {log_file}")
        click.echo()
        click.echo("Monitor with: python runpod_cli.py train check")
        click.echo("Stop with: python runpod_cli.py train stop")
    else:
        click.echo("Training failed to start. Check logs:", err=True)
        ssh_cmd(f"tail -50 {log_file}", check=False)
        sys.exit(1)


@grid.command("run-all")
@click.option("--rope-bases", type=str, default=None, help="Comma-separated RoPE base frequencies")
@click.option("--truncations", type=str, default=None, help="Comma-separated truncation splits")
@click.option("--replicates", type=int, default=3, help="Number of replicates per config")
@click.option("--single-gpu", is_flag=True, default=True, help="Use single GPU mode")
@click.option("--start-from", type=int, default=0, help="Start from config ID (for resuming)")
def cmd_grid_run_all(rope_bases: str | None, truncations: str | None, replicates: int, single_gpu: bool, start_from: int):
    """Run all grid search configurations sequentially on the remote instance."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    # Build grid_search.py command to get total config count
    # Global options must come before subcommand
    local_args = ["python", "grid_search.py", "--replicates", str(replicates)]
    if rope_bases:
        local_args.extend(["--rope-bases"] + rope_bases.split(","))
    if truncations:
        local_args.extend(["--truncations"] + truncations.split(","))
    local_args.append("list")

    result = run_cmd(local_args, cwd=REPO_DIR, capture=True)
    lines = result.stdout.strip().split("\n")

    # Count configs (lines starting with a number after the header)
    config_count = sum(1 for line in lines if line and line[0:3].strip().isdigit())

    click.echo(f"Will run {config_count - start_from} configurations (starting from {start_from})")
    click.echo("This will run sequentially - each run must complete before the next starts.")
    click.echo()

    for config_id in range(start_from, config_count):
        click.echo(f"\n{'='*60}")
        click.echo(f"Starting configuration {config_id + 1}/{config_count}")
        click.echo(f"{'='*60}")

        # Run this config
        ctx = click.Context(cmd_grid_run)
        ctx.invoke(cmd_grid_run, config_id=config_id, rope_bases=rope_bases,
                   truncations=truncations, replicates=replicates, single_gpu=single_gpu)

        # Wait for completion
        click.echo("\nWaiting for training to complete...")
        import time
        while True:
            time.sleep(30)
            result = ssh_cmd(f"test -f {REMOTE_PIDFILE} && ps -p $(cat {REMOTE_PIDFILE}) > /dev/null 2>&1 && echo running || echo stopped", capture=True)
            if result.stdout.strip() != "running":
                click.echo("Training completed.")
                break
            # Show progress
            ssh_cmd(f"tail -1 {REMOTE_REPO}/train.log 2>/dev/null || true", check=False)

    click.echo(f"\n{'='*60}")
    click.echo("All grid search configurations completed!")
    click.echo("Run 'python runpod_cli.py grid results' to see aggregated results.")


@grid.command("results")
@click.option("--csv", is_flag=True, help="Output as CSV")
@click.option("--json", is_flag=True, help="Output as JSON")
def cmd_grid_results(csv: bool, json_out: bool):
    """Fetch and display grid search results from remote."""
    if not ssh_available():
        click.echo("Error: Cannot connect to RunPod. Run 'configure' first.", err=True)
        sys.exit(1)

    # Sync logs back from remote
    click.echo("Fetching logs from remote...")
    rsync_args = [
        "rsync", "-avz", "--progress",
        "-e", "ssh",
        f"runpod:{REMOTE_REPO}/logs/",
        f"{REPO_DIR}/logs/"
    ]
    run_cmd(rsync_args, check=False)

    # Run local results aggregation
    args = ["python", "grid_search.py", "results"]
    if csv:
        args.append("--csv")
    if json_out:
        args.append("--json")
    run_cmd(args, cwd=REPO_DIR)


if __name__ == "__main__":
    cli()
