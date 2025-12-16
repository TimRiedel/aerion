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
# Run Prediction
# ============================================================================

python src/predict.py --config-name=execute_aerion \
    experiment_name=test_experiment \
    checkpoint.path=checkpoints/best.ckpt
