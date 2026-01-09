#!/bin/bash
#SBATCH --job-name=uaag2_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/uaag2_%j.out
#SBATCH --error=logs/uaag2_%j.err

# UAAG2 SLURM Training Script
# This script is designed to run on SLURM clusters with proper error handling

set -e  # Exit on error

# ==============================================================================
# SLURM Job Information
# ==============================================================================

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# ==============================================================================
# Configuration
# ==============================================================================

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

# Data Paths
DATA_PATH="/datasets/biochem/unaagi/debug_test.lmdb"
METADATA_PATH="/datasets/biochem/unaagi/unaagi_whole_v1.metadata.pkl"
DATA_INFO_PATH="/home/qcx679/hantang/UAAG2/data/full_graph/statistic.pkl"

# Experiment Configuration
EXPERIMENT_ID="uaag2_slurm_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
LOGGER_TYPE="wandb"  # Options: wandb, tensorboard
SAVE_DIR="./checkpoints"

# Sampler Configuration
USE_METADATA_SAMPLER=false
PDBBIND_WEIGHT=10.0

# ==============================================================================
# Environment Setup
# ==============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust for your cluster)
# module load cuda/11.8
# module load cudnn/8.6

# Check GPU availability
echo ""
echo "=========================================="
echo "GPU Assignment Information"
echo "=========================================="
echo "SLURM GPU allocation: $SLURM_GPUS_ON_NODE"
echo "SLURM GPU IDs: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "All GPUs on this node:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
    
    echo "GPUs assigned to this job:"
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        for gpu_id in ${CUDA_VISIBLE_DEVICES//,/ }; do
            nvidia-smi -i $gpu_id --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
        done
    else
        echo "CUDA_VISIBLE_DEVICES not set - all GPUs visible"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    fi
    echo ""
    
    echo "Full nvidia-smi output:"
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found!"
fi
echo "=========================================="

# Navigate to project directory
cd /home/qcx679/hantang/UAAG2 || exit 1

# Git information
echo ""
echo "Repository Information:"
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit: $(git rev-parse --short HEAD)"
fi
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate targetdiff

# Verify activation
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'targetdiff'"
    exit 1
fi

# Set library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Add src to PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Set OMP threads (prevent CPU oversubscription)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL settings for multi-GPU (if using multiple GPUs)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if causing issues

echo "Environment:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# ==============================================================================
# PyTorch/CUDA Diagnostics
# ==============================================================================

echo "=========================================="
echo "PyTorch and CUDA Check"
echo "=========================================="
python << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test GPU computation
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU computation test: PASSED")
    except Exception as e:
        print(f"✗ GPU computation test: FAILED - {e}")
        sys.exit(1)
else:
    print("ERROR: CUDA is not available!")
    print("This will cause the training script to fail.")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch/CUDA check failed!"
    exit 1
fi
echo "=========================================="
echo ""

# ==============================================================================
# Training Command
# ==============================================================================

# Determine GPU setting
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
echo "Number of GPUs detected: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected, but SLURM allocated GPU resources."
    echo "This indicates a configuration problem."
    exit 1
fi

# Build metadata sampler flag
if [ "$USE_METADATA_SAMPLER" = true ]; then
    SAMPLER_FLAG="--use-metadata-sampler"
else
    SAMPLER_FLAG="--no-metadata-sampler"
fi

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Experiment ID: $EXPERIMENT_ID"
echo "Batch size: $BATCH_SIZE"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Save directory: $SAVE_DIR"
echo "=========================================="
echo ""

# Run training with proper error handling
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

TRAIN_EXIT_CODE=$?

# ==============================================================================
# Cleanup and Summary
# ==============================================================================

echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Status: ✓ SUCCESS"
else
    echo "Status: ✗ FAILED"
fi
echo "=========================================="

exit $TRAIN_EXIT_CODE
