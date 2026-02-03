#!/bin/bash
#SBATCH --job-name=UAAG_train_16gpu
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1                    # Request 1 node for 8 GPUs total [1]
#SBATCH --gpus-per-node=8            # 8 GPU GCDs per node [1]
#SBATCH --ntasks-per-node=8          # 1 task per GPU GCD [2]
#SBATCH --cpus-per-task=7            # 56 available cores / 8 tasks per node [4]
#SBATCH --mem=480G                   # Standard memory for a full LUMI-G node [2]
#SBATCH --time=2-00:00:00
#SBATCH -o logs/train_%j_8GPUs0201.log
#SBATCH -e logs/train_%j.log


DATA_PATH=/scratch/project_465002574/unaagi_whole_v1.lmdb
MODEL=UAAG_model_official_8
DATA_INFO_PATH=/flash/project_465002574/UAAG2_main/data/statistic.pkl
METADATA_PATH=/scratch/project_465002574/unaagi_whole_v1.metadata.pkl

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "→ Setting up environment..."

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

export WORK_DIR=/flash/project_465002574/UAAG2_main
cd ${WORK_DIR}
mkdir -p logs

# --- Critical Stability & Performance Fixes ---
export HSA_ENABLE_SDMA=0             # Proven stability fix for MI250X [7-9]
export AMD_DIRECT_DISPATCH=0         # Proven stability fix [8, 9]
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 # Use Slingshot 11 interconnect [10, 11]
export NCCL_NET_GDR_LEVEL=3          # Enable GPU Direct RDMA [10, 11]

# # Additional stability fixes for dataloader issues
# export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
# export OMP_NUM_THREADS=7             # Match cpus-per-task to avoid oversubscription
# export OPENBLAS_NUM_THREADS=1        # Prevent nested parallelism

# # MIOpen cache must be on /tmp to avoid Lustre file-locking issues [10, 12]
# export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
# export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# # CPU Binding Mask to map CPU chiplets to GPU dies correctly [2, 13]
# c=fe
# MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"


# Activate your environment (adjust path as needed)
# If using conda-containerize or singularity:
# singularity exec --bind /flash:/flash container.sif bash -c "..."
# For now, assuming uv-based environment from hantang_env
cd ${WORK_DIR}

# Switch to prior_condition branch
echo ""
echo "→ Switching to prior_condition branch..."
git checkout prior_condition

# Check GPU
echo ""
echo "→ Checking GPU availability..."
rocm-smi || echo "Warning: rocm-smi not available"

mkdir -p logs
mkdir -p $MIOPEN_USER_DB_PATH

python scripts/run_train.py \
  --logger-type wandb \
  --batch-size 8 --test-interval 5 \
  --gpus 8 --mask-rate 0 --test-size 32 --train-size 0.99 \
  --num_nodes 1 \
  --id Full_mask_8_gpu_${MODEL}_0203_prior --max-virtual-nodes 5 --use_metadata_sampler \
  --training_data $DATA_PATH \
  --data_info_path $DATA_INFO_PATH \
  --metadata_path $METADATA_PATH --num-workers 4