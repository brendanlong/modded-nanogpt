# RunPod Setup Guide

Notes on running modded-nanogpt on RunPod, based on testing with a single H100 PCIe instance.

## SSH Configuration (Local Machine)

Set up SSH multiplexing for faster, more reliable connections. Add this to your `~/.ssh/config`:

```
Host runpod
    HostName <your-runpod-ip>
    Port <your-runpod-port>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then create the sockets directory:
```bash
mkdir -p ~/.ssh/sockets
```

This allows:
- Reusing SSH connections (faster subsequent commands)
- Keeping connections alive for 10 minutes after last use
- Automatic reconnection on network blips

After setup, connect with just `ssh runpod` instead of the full command.

**Note**: RunPod assigns a new port each time you start/restart an instance, so you'll need to update the Port in your config.

## Instance Configuration

### GPU Selection
- **H100 PCIe** (80GB) works well
- H100 SXM would also work

### Volume Configuration
- **Root disk**: 20GB is sufficient (code and packages fit)
- **Workspace volume**: The `/workspace` directory is a persistent network volume with ample space (hundreds of TB shared). Use this for:
  - Cloning the repo
  - Storing training data (~20GB for full dataset)
  - Log files

The training data downloads to `data/fineweb10B/` relative to the repo, so clone to `/workspace` to ensure enough space.

## Initial Setup

### 1. Install tmux (recommended)

The base image doesn't include tmux. Install it for persistent sessions:

```bash
apt-get update && apt-get install -y tmux
```

Using tmux is highly recommended - SSH connections can drop, and you don't want to lose a training run.

### 2. Clone and Setup

```bash
cd /workspace
git clone https://github.com/KellerJordan/modded-nanogpt.git
cd modded-nanogpt
git checkout <your-branch>  # if testing a branch
```

### 3. Install Dependencies

The base RunPod PyTorch image has an older PyTorch. You need the nightly:

```bash
# Install requirements (includes the 'kernels' package for Flash Attention 3)
pip install -r requirements.txt

# Install PyTorch nightly (REQUIRED - older PyTorch lacks necessary Triton features)
pip install torch==2.10.0.dev20251210+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126
```

The nightly PyTorch install takes a few minutes and will show dependency warnings about torchaudio/torchvision - these can be ignored.

### 4. Download Training Data

```bash
# Download full dataset (~20GB, 103 shards)
python data/cached_fineweb10B.py

# Or download partial dataset for faster testing (e.g., 9 shards)
python data/cached_fineweb10B.py 9
```

## Running Training

### Start a tmux Session

```bash
tmux new-session -s train
cd /workspace/modded-nanogpt
```

### Run Training

```bash
# Single GPU mode
SINGLE_GPU=1 torchrun --nproc_per_node=1 train_gpt.py

# Full 8-GPU mode (if you have 8 GPUs)
torchrun --nproc_per_node=8 train_gpt.py
```

### Detach and Reattach

- Detach from tmux: `Ctrl+B` then `D`
- Reattach: `tmux attach -t train`
- List sessions: `tmux ls`

## Monitoring

### Check Training Progress

From outside tmux:
```bash
tmux capture-pane -t train -p -S -20  # Show last 20 lines
```

### Check GPU Utilization

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

Expected during training:
- GPU utilization: ~99%
- Memory used: ~75GB (single GPU mode)

### Check Logs

Logs are written to `logs/<run-id>.txt` with full training output.

## Troubleshooting

### "No module named 'triton.tools.tensor_descriptor'"
You're using an older PyTorch/Triton. Install the nightly PyTorch as shown above.

### "No module named 'kernels'"
Run `pip install -r requirements.txt` to install the Flash Attention 3 kernels package.

### OOM Errors
If using the full batch size on a single GPU, you'll get OOM errors. Use `SINGLE_GPU=1` which configures appropriate batch sizes.

### SSH Connection Drops
This is why tmux is essential. Your training will continue in the tmux session even if SSH disconnects. Just reconnect and `tmux attach -t train`.

### Instance Restarts
If the RunPod instance restarts:
- The `/workspace` volume persists (your repo and data are safe)
- You'll need to reinstall tmux and may need to reinstall PyTorch nightly
- The torch.compile cache may need to rebuild (~7 min warmup)

## Cost Optimization

- H100 PCIe instances are typically $3-4/hour
- A full single-GPU training run takes ~15 minutes (including first-time compile)
- Subsequent runs take ~8 minutes (compile cached)
- Stop the instance when not in use - the workspace volume persists
