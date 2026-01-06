#!/bin/bash

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================================================
# Environment Setup
# ============================================================================

export OMP_NUM_THREADS=4
ulimit -n 8192


# ============================================================================
# Run Training
# ============================================================================
# PyTorch Lightning automatically handles single-GPU and multi-GPU training.
# Set trainer.devices="auto" (default) to use all available GPUs, or specify
# a number like trainer.devices=4 to use 4 GPUs, or trainer.devices=1 for single GPU.

python src/main.py --config-name=execute_transformer \
    stage=train \
    experiment_name=transformer-debug \
    dataset=single_day \
    dataloader.batch_size=128 \
    trainer.max_epochs=1 \
