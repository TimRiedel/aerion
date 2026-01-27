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

python src/main.py --config-name=execute_aerion \
    stage=train \
    dataset=single_day \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=1 \
    contexts.flightinfo.enabled=false \
    loss.weights.mse=0.0 \
    loss.weights.ade=0.5 \
    loss.weights.fde=0.5 \
    loss.weights.alignment=0.2 \
    loss.alignment_scale_factor=10000.0 \
    scheduled_sampling.enabled=true \

    # execution.num_waypoints_to_predict=10 \
    # trainer.fast_dev_run=True \
    # execution.num_trajectories_to_predict=1 \
