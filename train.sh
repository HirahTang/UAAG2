#!/bin/bash

# UAAG2 Training Script
# Updated for new modular structure with uv environment support

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

# Training Hyperparameters
BATCH_SIZE=8
NUM_EPOCHS=5000
TRAIN_SIZE=0.99
TEST_SIZE=32
LEARNING_RATE=5e-4

# Model Configuration
MAX_VIRTUAL_NODES=5
MASK_RATE=0.0
NUM_LAYERS=7

# Data Paths (adjust these to your system)
DATA_PATH="/datasets/biochem/unaagi/debug_test.lmdb"
METADATA_PATH="/datasets/biochem/unaagi/unaagi_whole_v1.metadata.pkl"
DATA_INFO_PATH="/home/qcx679/hantang/UAAG2/data/full_graph/statistic.pkl"

# Experiment Configuration
EXPERIMENT_ID="uaag2_training_$(date +%Y%m%d_%H%M%S)"
LOGGER_TYPE="wandb"  # Options: wandb, tensorboard
SAVE_DIR="./checkpoints"

# Sampler Configuration
USE_METADATA_SAMPLER=false  # Set to false to use RandomSampler instead
PDBBIND_WEIGHT=10.0

# ==============================================================================
# Environment Setup
# ==============================================================================

echo "=========================================="
echo "UAAG2 Training Script"
echo "=========================================="
echo "Experiment ID: $EXPERIMENT_ID"
echo "Start time: $(date)"
echo "=========================================="

# Check if running on SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Node: $SLURMD_NODENAME"
    echo "Hostname: $(hostname)"
fi

# GPU Information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Repository Information
echo "Repository Information:"
echo "Working directory: $(pwd)"
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit: $(git rev-parse --short HEAD)"
    echo "Status: $(git status --porcelain | wc -l) uncommitted changes"
fi
echo "=========================================="

# Activate environment
# Uncomment the appropriate method for your setup:

# Method 1: uv managed environment (recommended)
if [ -d ".venv" ]; then
    echo "Activating uv virtual environment..."
    source .venv/bin/activate
    export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
fi

# Method 2: conda environment (legacy)
# source ~/.bashrc
# conda activate uaag2
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Add src directory to PYTHONPATH so uaag2 package can be found
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ==============================================================================
# Training Command
# ==============================================================================

# Build metadata sampler flag
if [ "$USE_METADATA_SAMPLER" = true ]; then
    SAMPLER_FLAG="--use-metadata-sampler"
else
    SAMPLER_FLAG="--no-metadata-sampler"
fi

echo "Starting training..."
echo "Command:"
echo "python src/uaag2/train.py \\"
echo "    --logger-type $LOGGER_TYPE \\"
echo "    --batch-size $BATCH_SIZE \\"
echo "    --gpus $NUM_GPUS \\"
echo "    --num-epochs $NUM_EPOCHS \\"
echo "    --train-size $TRAIN_SIZE \\"
echo "    --test-size $TEST_SIZE \\"
echo "    --lr $LEARNING_RATE \\"
echo "    --mask-rate $MASK_RATE \\"
echo "    --max-virtual-nodes $MAX_VIRTUAL_NODES \\"
echo "    --num-layers $NUM_LAYERS \\"
echo "    --pdbbind-weight $PDBBIND_WEIGHT \\"
echo "    --training_data $DATA_PATH \\"
echo "    --metadata_path $METADATA_PATH \\"
echo "    --data_info_path $DATA_INFO_PATH \\"
echo "    --save-dir $SAVE_DIR \\"
echo "    --id $EXPERIMENT_ID \\"
echo "    $SAMPLER_FLAG"
echo "=========================================="
echo ""

python src/uaag2/train.py \
    --logger-type "$LOGGER_TYPE" \
    --batch-size "$BATCH_SIZE" \
    --gpus "$NUM_GPUS" \
    --num-epochs "$NUM_EPOCHS" \
    --train-size "$TRAIN_SIZE" \
    --test-size "$TEST_SIZE" \
    --lr "$LEARNING_RATE" \
    --mask-rate "$MASK_RATE" \
    --max-virtual-nodes "$MAX_VIRTUAL_NODES" \
    --num-layers "$NUM_LAYERS" \
    --pdbbind-weight "$PDBBIND_WEIGHT" \
    --training_data "$DATA_PATH" \
    --metadata_path "$METADATA_PATH" \
    --data_info_path "$DATA_INFO_PATH" \
    --save-dir "$SAVE_DIR" \
    --id "$EXPERIMENT_ID" \
    $SAMPLER_FLAG

# ==============================================================================
# Cleanup
# ==============================================================================

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
