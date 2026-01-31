# Single GPU Mode

This document describes how to run modded-nanogpt on a single H100 GPU for testing and validation before committing to a full 8-GPU speedrun.

## Overview

Single GPU mode uses the same effective batch size as the 8-GPU run by processing 1/2 of the total batch with 2 gradient accumulation steps. This maintains token efficiency (same loss at same step count) while achieving ~99% GPU utilization on an 80GB H100.

## Usage

```bash
SINGLE_GPU=1 torchrun --nproc_per_node=1 train_gpt.py
```

## How It Works

| Config | Microbatch Size | Grad Accum | Effective Batch | GPU Util | VRAM | Step Time |
|--------|-----------------|------------|-----------------|----------|------|-----------|
| 8-GPU (original) | 1/8 each | 1 per GPU | Full | ~99% | ~30GB | ~100ms |
| Single GPU | 1/2 | 2 | Full | ~99% | ~63GB | ~375ms |

The key insight is that the per-GPU batch in the 8-GPU run (1/8 of total) only achieves ~22% GPU utilization on a single H100. By using 4x larger microbatches with fewer accumulation steps, we saturate the GPU while maintaining the same total tokens per optimizer step.

## Expected Performance

- **Warmup/compile time**: ~7 minutes (first run only, cached afterward)
- **Training time**: ~10 minutes for 1600 steps
- **Total time**: ~17 minutes first run, ~10 minutes subsequent runs
- **Average step time**: ~375ms

For comparison, the 8-GPU speedrun completes in under 2 minutes, so single GPU mode is roughly 5x slower wall-clock time.

## Validation Checkpoints

Expected validation loss progression (from actual test run):

| Step | Val Loss |
|------|----------|
| 0 | 10.83 |
| 250 | 4.82 |
| 500 | 4.48 |
| 750 | 4.00 |
| 1000 | 3.78 |
| 1250 | 3.54 |
| 1500 | 3.45 |
| 1600 | 3.42 |

Note: The test run achieved 3.42 loss vs the 3.28 target. This may be due to using a partial dataset (60 of 103 training shards). With the full dataset, results should match the 8-GPU run more closely.

## Hardware Requirements

- **GPU**: NVIDIA H100 (80GB) recommended
  - H100 PCIe or SXM both work
  - A100 80GB may work but requires `DISABLE_FP8=1` (untested)
- **VRAM**: ~75GB peak usage
- **Disk**: ~25GB for code + data

## Limitations

- Flash Attention 3 requires Hopper architecture (H100)
- FP8 matmuls require H100's native FP8 support
- The nightly PyTorch version is required for Triton compatibility
