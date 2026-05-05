#!/bin/bash
#SBATCH --job-name=UAAG_train_16gpu
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=2                    # Request 2 nodes for 16 GPUs total [1, 2]
#SBATCH --gpus-per-node=8            # 8 GPU GCDs per node [1, 3, 4]
#SBATCH --ntasks-per-node=8          # 1 process per GPU [5, 6]
#SBATCH --cpus-per-task=7            # 56 available cores / 8 tasks per node [7-9]
#SBATCH --mem=480G                   # Standard memory for a full node [5, 10, 11]
#SBATCH --time=2-00:00:00
#SBATCH -o logs/train_16gpu_%j.log
#SBATCH -e logs/train_16gpu_%j.log

# ============================================================================
# PATHS & DATA SETUP
# ============================================================================
DATA_PATH=/scratch/project_465002574/unaagi_whole_v1.lmdb
MODEL=UAAG_model_official_8
DATA_INFO_PATH=/flash/project_465002574/UAAG2_main/data/statistic.pkl
METADATA_PATH=/scratch/project_465002574/unaagi_whole_v1.metadata.pkl
WORK_DIR=/flash/project_465002574/UAAG2_main

# ============================================================================
# DISTRIBUTED ARCHITECTURE SETUP
# ============================================================================
# 1. Dynamically identify the master node for PyTorch rendezvous [12-14]
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# 2. CPU Binding Mask ("Green Ring" topology)
# This hexadecimal mask pins each task to the CPU CCD closest to its GPU [10, 15-17]
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

# Networking Fixes for Slingshot Interconnect [18-21]
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 # Use high-speed network interfaces
export NCCL_NET_GDR_LEVEL=3                  # Enable GPU Direct RDMA

# MIOpen Cache Redirection to /tmp (Mandatory to avoid Lustre deadlocks) [22-25]
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Prep /tmp on both nodes
srun mkdir -p $MIOPEN_USER_DB_PATH

cd ${WORK_DIR}

# ============================================================================
# EXECUTION (using srun for 16 parallel tasks)
# ============================================================================
srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    
    python scripts/run_train.py \
      --logger-type wandb \
      --batch-size 8 \
      --test-interval 5 \
      --gpus 16 \
      --num_nodes 2 \
      --id Full_mask_16_gpu_${MODEL}_0322_DDP_test \
      --max-virtual-nodes 5 \
      --use_metadata_sampler \
      --training_data $DATA_PATH \
      --data_info_path $DATA_INFO_PATH \
      --metadata_path $METADATA_PATH \
      --num-workers 4 \
      --load-ckpt /flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_8_gpu_UAAG_model_official_8_0202/last.ckpt
"

# Cleanup /tmp
srun rm -rf $MIOPEN_USER_DB_PATH