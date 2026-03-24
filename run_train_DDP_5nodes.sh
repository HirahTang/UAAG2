#!/bin/bash
#SBATCH --job-name=UAAG_train_40gpu
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=5
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/train_40gpu_%j.log
#SBATCH -e logs/train_40gpu_%j.log

# 1. DIRECTORY SETUP
OUTPUT_DIR=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/Full_mask_gpu_UAAG_model_condition_ProteinMPNN_128_0324
TRAINING_DATA_PATH=/scratch/project_465002574/PDB/uaag2_eqgat_lmdb_shards
# TRAINING_DATA_PATH can point to:
# - a single LMDB file, or
# - a directory that contains many *.lmdb shard files
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# 2. RENDEZVOUS SETUP
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29522  # Fresh port to avoid lingering EADDRINUSE

# 3. CPU BINDING (Green Ring topology)
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# 4. ENVIRONMENT SETUP
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

# Networking & MIOpen stability fixes
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
srun mkdir -p $MIOPEN_USER_DB_PATH

# 5. EXECUTION
# We treat each task as a virtual node (total 40) because of GPU isolation
srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export NODE_RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=0
    
    python scripts/run_train.py \
      --gpus 1 \
      --num_nodes 40 \
      --batch-size 8 \
      --logger-type wandb \
      --id Full_mask_gpu_UAAG_model_condition_ProteinMPNN_128_0324 \
      --training_data $TRAINING_DATA_PATH \
      --use_protein_mpnn_context_128 \
      --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl \
      --num-workers 4
"

srun rm -rf $MIOPEN_USER_DB_PATH