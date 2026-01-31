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
