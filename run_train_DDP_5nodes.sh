#!/bin/bash
#SBATCH --job-name=UAAG_train_40gpu
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=5                    # Request 5 nodes for 40 GCDs total [1, 2]
#SBATCH --gpus-per-node=8            # 8 GCDs per node on LUMI-G [1, 3]
#SBATCH --ntasks-per-node=8          # One independent process per GPU [2, 4]
#SBATCH --cpus-per-task=7            # Match the 1/8 rule for resource balance [2, 5]
#SBATCH --mem=480G                   # Standard memory for a full LUMI-G node [2, 6]
#SBATCH --time=2-00:00:00
#SBATCH -o logs/train_40gpu_%j.log
#SBATCH -e logs/train_40gpu_%j.log

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
# Dynamically identify the lead node for PyTorch rendezvous [7, 8]
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# CPU Binding Mask ("Linear assignment" for MI250X) [9, 10]
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

# Networking & Stability Fixes [11, 12]
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 # Force high-speed interconnect [12, 13]
export NCCL_NET_GDR_LEVEL=3                  # Enable GPU Direct RDMA [12, 13]

# MIOpen Cache Redirection (Mandatory to avoid Lustre deadlocks) [11, 14]
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
srun mkdir -p $MIOPEN_USER_DB_PATH

cd ${WORK_DIR}

# ============================================================================
# EXECUTION (using srun to launch 40 parallel tasks)
# ============================================================================
# ROCR_VISIBLE_DEVICES isolates each process so it only sees 1 GPU [15, 16]
# Python --gpus is now set to 1 because each task only manages 1 local GPU [17, 18]

srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    
    python scripts/run_train.py \
      --logger-type wandb \
      --batch-size 8 \
      --test-interval 5 \
      --gpus 1 \
      --num_nodes 5 \
      --id Full_mask_40_gpu_${MODEL}_0202 \
      --max-virtual-nodes 5 \
      --use_metadata_sampler \
      --training_data $DATA_PATH \
      --data_info_path $DATA_INFO_PATH \
      --metadata_path $METADATA_PATH \
      --num-workers 4 \
      --load-ckpt /flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_8_gpu_UAAG_model_official_8_0202/last.ckpt
"

# Cleanup local /tmp cache
srun rm -rf $MIOPEN_USER_DB_PATH