#!/bin/bash
#SBATCH --job-name=UAAG_demo_4gpu
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH -o logs/train_demo_4gpu_%j.log
#SBATCH -e logs/train_demo_4gpu_%j.log

set -euo pipefail

# 1. DIRECTORY / RUN CONFIG
OUTPUT_DIR=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runDemo_4_gpu_UAAG_model
TRAINING_DATA_PATH=/scratch/project_465002574/PDB/uaag2_eqgat_lmdb_shards
DATA_INFO_PATH=/flash/project_465002574/UAAG2_main/data/statistic.pkl
RUN_ID=Demo_4_gpu_UAAG_model

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# 2. RENDEZVOUS
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29524

# 3. CPU BINDING (first 4 masks used)
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000"

# 4. ENVIRONMENT
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
srun mkdir -p "$MIOPEN_USER_DB_PATH"

export WORK_DIR=/flash/project_465002574/UAAG2_main
cd "$WORK_DIR"

echo "Running demo training on 1 node / 4 GPUs"
echo "TRAINING_DATA_PATH=$TRAINING_DATA_PATH"
echo "RUN_ID=$RUN_ID"

# 5. EXECUTION
# One task controls one GPU; treat tasks as virtual nodes (4 total).
srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export NODE_RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=0

    python scripts/run_train.py \
      --gpus 1 \
      --num_nodes 4 \
      --batch-size 8 \
      --num-epochs 10 \
      --logger-type wandb \
      --id $RUN_ID \
      --training_data $TRAINING_DATA_PATH \
      --data_info_path $DATA_INFO_PATH \
      --metadata_path $METADATA_PATH \
      --num-workers 4 \
      --test-size 32 \
      --train-size 0.99 \
      --mask-rate 0 \
      --max-virtual-nodes 5 \
      --use_protein_mpnn_context_128
"

srun rm -rf "$MIOPEN_USER_DB_PATH"
