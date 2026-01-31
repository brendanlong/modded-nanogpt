# Claude Code Guide

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## RunPod CLI Tool

A Python CLI for managing training sessions on RunPod (`runpod_cli.py`):

| Command | Description |
|---------|-------------|
| `python runpod_cli.py ssh-keygen` | Generate SSH key and show setup instructions |
| `python runpod_cli.py configure '<connection-string>'` | Parse RunPod connection string, update SSH config |
| `python runpod_cli.py sync-code` | Rsync code to remote (excludes .git, data, logs) |
| `python runpod_cli.py fetch-data [--shards N]` | Download training data on remote |
| `python runpod_cli.py setup` | Full setup: tmux, sync, pip install, PyTorch nightly |
| `python runpod_cli.py train start [--single-gpu]` | Start training in background with nohup |
| `python runpod_cli.py train check [-n LINES] [-f]` | Check status and show logs |
| `python runpod_cli.py train stop` | Stop running training |
| `python runpod_cli.py gpu-status` | Show nvidia-smi |
| `python runpod_cli.py shell` | Open tmux session on remote |

See `RUNPOD_SETUP.md` for detailed RunPod setup instructions and `SINGLE_GPU.md` for single GPU mode documentation.

## Grid Search

Grid search for RoPE base frequency and truncation split hyperparameters.

### Local Commands

```bash
# List all configurations
python grid_search.py list

# Custom grid values
python grid_search.py --rope-bases 512 1024 2048 --truncations 0.25 0.5 0.75 --replicates 3 list

# View results (after runs complete)
python grid_search.py results
python grid_search.py results --csv
```

### Remote (RunPod) Commands

```bash
# List configurations
python runpod_cli.py grid list

# Run a specific configuration
python runpod_cli.py grid run 0

# Run all configurations sequentially
python runpod_cli.py grid run-all

# Resume from a specific config (after interruption)
python runpod_cli.py grid run-all --start-from 5

# Fetch results from remote
python runpod_cli.py grid results
```

### Custom Grid Values

Use comma-separated values for custom grids:

```bash
python runpod_cli.py grid list --rope-bases 768,1024,1536 --truncations 0.4,0.5,0.6 --replicates 2
python runpod_cli.py grid run-all --rope-bases 768,1024,1536 --truncations 0.4,0.5,0.6 --replicates 2
```

### Files

- `grid_search.py` - Grid search configuration and results aggregation
- `train_gpt_grid.py` - Training script with configurable RoPE parameters via environment variables
