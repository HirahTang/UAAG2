#!/bin/bash
#SBATCH --job-name=UAAG_train
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/train_%j.log
#SBATCH -e logs/train_%j.log

DATA_PATH=/scratch/project_465002574/debug_test.lmdb
MODEL=UAAG_model
DATA_INFO_PATH=/flash/project_465002574/UAAG2_main/data/statistic.pkl
NUM_SAMPLES=100  # Reduced for testing
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15
SPLIT_INDEX=0
PROTEIN_ID=ENVZ_ECOLI

WORK_DIR=/flash/project_465002574/UAAG2_main

mkdir -p logs
# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "→ Setting up environment..."

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

# Activate your environment (adjust path as needed)
# If using conda-containerize or singularity:
# singularity exec --bind /flash:/flash container.sif bash -c "..."
# For now, assuming uv-based environment from hantang_env
cd ${WORK_DIR}

# Check GPU
echo ""
echo "→ Checking GPU availability..."
rocm-smi || echo "Warning: rocm-smi not available"

# ============================================================================
# STEP 1: SAMPLING
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 1: Running Training"
echo "============================================================================"
echo "[$(date)] Starting training ..."
python scripts/run_train.py --logger-type wandb --batch-size 8 \
    --test-interval 5 --gpus 1 --mask-rate 0 --test-size 32 --train-size 0.99 \
    --id Full_mask_5_virtual_node_mask_token_atomic_only_mask_diffusion_LUMI_test --max-virtual-nodes 5 --num-epochs 5000 \
    --training_data /scratch/project_465002574/debug_test.lmdb --data_info_path ${DATA_INFO_PATH}