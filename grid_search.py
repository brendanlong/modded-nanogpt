#!/usr/bin/env python3
"""
Grid search for RoPE base frequency and truncation split hyperparameters.

Usage:
    python grid_search.py list                    # Show all configurations
    python grid_search.py run <config_id>         # Run a specific configuration
    python grid_search.py results                 # Aggregate and display results
    python grid_search.py results --csv           # Export results as CSV

Environment variables:
    SINGLE_GPU=1    Run in single GPU mode
    DATA_PATH=...   Path to data directory
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional


@dataclass
class GridConfig:
    """Configuration for a single grid search point."""
    rope_base: float
    truncation: float
    replicate: int

    @property
    def config_id(self) -> str:
        """Unique identifier for this configuration."""
        return f"rope{int(self.rope_base)}_trunc{self.truncation:.2f}_rep{self.replicate}"

    def to_env(self) -> dict:
        """Return environment variables for this configuration."""
        return {
            "ROPE_BASE": str(self.rope_base),
            "TRUNCATION_SPLIT": str(self.truncation),
            "GRID_CONFIG_ID": self.config_id,
        }


# Default grid values - can be overridden via command line
DEFAULT_ROPE_BASES = [512, 1024, 2048, 4096]
DEFAULT_TRUNCATIONS = [0.25, 0.5, 0.75]
DEFAULT_REPLICATES = 3


def generate_configs(
    rope_bases: list[float] = None,
    truncations: list[float] = None,
    replicates: int = None,
) -> list[GridConfig]:
    """Generate all configurations for the grid search."""
    rope_bases = rope_bases or DEFAULT_ROPE_BASES
    truncations = truncations or DEFAULT_TRUNCATIONS
    replicates = replicates or DEFAULT_REPLICATES

    configs = []
    for rope_base, truncation in product(rope_bases, truncations):
        for rep in range(replicates):
            configs.append(GridConfig(rope_base, truncation, rep))
    return configs


def parse_log_file(log_path: Path) -> Optional[dict]:
    """Parse a training log file to extract final validation loss and timing."""
    if not log_path.exists():
        return None

    content = log_path.read_text()

    # Find final validation loss (last val_loss entry)
    val_losses = re.findall(r"step:(\d+)/\d+ val_loss:([\d.]+)", content)
    if not val_losses:
        return None

    final_step, final_loss = val_losses[-1]

    # Get all validation losses for the curve
    loss_curve = [(int(step), float(loss)) for step, loss in val_losses]

    # Get training time
    time_match = re.search(r"train_time:(\d+)ms", content)
    train_time_ms = int(time_match.group(1)) if time_match else None

    return {
        "final_step": int(final_step),
        "final_loss": float(final_loss),
        "loss_curve": loss_curve,
        "train_time_ms": train_time_ms,
    }


def find_log_for_config(config: GridConfig, logs_dir: Path = Path("logs")) -> Optional[Path]:
    """Find the log file for a given configuration."""
    # Look for logs with the config ID in the filename or content
    if not logs_dir.exists():
        return None

    for log_file in logs_dir.glob("*.txt"):
        content = log_file.read_text()
        if f"GRID_CONFIG_ID={config.config_id}" in content:
            return log_file

    return None


def list_configs(args):
    """List all grid search configurations."""
    configs = generate_configs(
        rope_bases=args.rope_bases,
        truncations=args.truncations,
        replicates=args.replicates,
    )

    print(f"Grid search: {len(configs)} configurations")
    print(f"  RoPE bases: {args.rope_bases or DEFAULT_ROPE_BASES}")
    print(f"  Truncations: {args.truncations or DEFAULT_TRUNCATIONS}")
    print(f"  Replicates: {args.replicates or DEFAULT_REPLICATES}")
    print()
    print("ID  Config ID                          RoPE Base  Truncation  Rep")
    print("-" * 70)

    for i, config in enumerate(configs):
        print(f"{i:3d} {config.config_id:35s} {config.rope_base:9.0f}  {config.truncation:10.2f}  {config.replicate:3d}")


def run_config(args):
    """Run a specific grid search configuration."""
    configs = generate_configs(
        rope_bases=args.rope_bases,
        truncations=args.truncations,
        replicates=args.replicates,
    )

    if args.config_id >= len(configs):
        print(f"Error: config_id {args.config_id} out of range (0-{len(configs)-1})")
        sys.exit(1)

    config = configs[args.config_id]
    print(f"Running configuration: {config.config_id}")
    print(f"  RoPE base: {config.rope_base}")
    print(f"  Truncation: {config.truncation}")
    print(f"  Replicate: {config.replicate}")

    # Build environment
    env = os.environ.copy()
    env.update(config.to_env())

    # Use the grid-enabled training script
    script = "train_gpt_grid.py"
    if not Path(script).exists():
        print(f"Error: {script} not found. Run grid search setup first.")
        sys.exit(1)

    # Determine number of GPUs
    if env.get("SINGLE_GPU") == "1":
        cmd = ["torchrun", "--standalone", "--nproc_per_node=1", script]
    else:
        # Multi-GPU mode
        nproc = args.nproc or 8
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", script]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


def aggregate_results(args):
    """Aggregate and display grid search results."""
    configs = generate_configs(
        rope_bases=args.rope_bases,
        truncations=args.truncations,
        replicates=args.replicates,
    )

    logs_dir = Path(args.logs_dir)
    results = []

    for config in configs:
        log_path = find_log_for_config(config, logs_dir)
        if log_path:
            parsed = parse_log_file(log_path)
            if parsed:
                results.append({
                    "config_id": config.config_id,
                    "rope_base": config.rope_base,
                    "truncation": config.truncation,
                    "replicate": config.replicate,
                    "log_file": str(log_path),
                    **parsed,
                })

    if not results:
        print("No results found.")
        return

    if args.csv:
        # CSV output
        print("config_id,rope_base,truncation,replicate,final_loss,train_time_ms")
        for r in results:
            print(f"{r['config_id']},{r['rope_base']},{r['truncation']},{r['replicate']},{r['final_loss']},{r.get('train_time_ms', '')}")
        return

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Aggregate by (rope_base, truncation)
    from collections import defaultdict
    import statistics

    grouped = defaultdict(list)
    for r in results:
        key = (r["rope_base"], r["truncation"])
        grouped[key].append(r["final_loss"])

    print("Aggregated Results (mean ± std)")
    print("=" * 60)
    print(f"{'RoPE Base':>10} {'Truncation':>12} {'Loss':>12} {'N':>5}")
    print("-" * 60)

    # Sort by loss (ascending)
    sorted_results = []
    for (rope_base, truncation), losses in grouped.items():
        mean_loss = statistics.mean(losses)
        std_loss = statistics.stdev(losses) if len(losses) > 1 else 0
        sorted_results.append((mean_loss, std_loss, rope_base, truncation, len(losses)))

    sorted_results.sort()

    for mean_loss, std_loss, rope_base, truncation, n in sorted_results:
        if n > 1:
            loss_str = f"{mean_loss:.4f}±{std_loss:.4f}"
        else:
            loss_str = f"{mean_loss:.4f}"
        print(f"{rope_base:>10.0f} {truncation:>12.2f} {loss_str:>12} {n:>5}")

    print()
    print(f"Total configurations: {len(configs)}")
    print(f"Completed runs: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Grid search for RoPE hyperparameters")

    # Global options
    parser.add_argument("--rope-bases", type=float, nargs="+",
                       help=f"RoPE base frequencies (default: {DEFAULT_ROPE_BASES})")
    parser.add_argument("--truncations", type=float, nargs="+",
                       help=f"Truncation splits (default: {DEFAULT_TRUNCATIONS})")
    parser.add_argument("--replicates", type=int,
                       help=f"Number of replicates per config (default: {DEFAULT_REPLICATES})")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List all configurations")
    list_parser.set_defaults(func=list_configs)

    # run command
    run_parser = subparsers.add_parser("run", help="Run a specific configuration")
    run_parser.add_argument("config_id", type=int, help="Configuration ID (from 'list' command)")
    run_parser.add_argument("--nproc", type=int, help="Number of GPUs (default: 8 or 1 for single-gpu)")
    run_parser.set_defaults(func=run_config)

    # results command
    results_parser = subparsers.add_parser("results", help="Aggregate and display results")
    results_parser.add_argument("--logs-dir", default="logs", help="Directory containing log files")
    results_parser.add_argument("--csv", action="store_true", help="Output as CSV")
    results_parser.add_argument("--json", action="store_true", help="Output as JSON")
    results_parser.set_defaults(func=aggregate_results)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
