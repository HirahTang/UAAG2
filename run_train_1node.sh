#!/bin/bash
#SBATCH --job-name=UAAG_train_8gpu
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2:00:00
#SBATCH -o logs/train_1node_%j.log
#SBATCH -e logs/train_1node_%j.log

# =============================================================================
# Dataset source switches — set to 1 to enable each source
# =============================================================================
USE_LMDB=0
USE_PDB=1
USE_PDBBIND=1
USE_NCAA=1

LMDB_WEIGHT=1.0
PDB_WEIGHT=1.0
PDBBIND_WEIGHT=1.0
NCAA_WEIGHT=1.0

# =============================================================================
# Paths
# =============================================================================
OUTPUT_DIR=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/train_1node_test
LMDB_DATA=/scratch/project_465002574/PDB/uaag2_eqgat_lmdb_shards
PDB_DIR=/scratch/project_465002574/PDB/PDB_processed
LATENT_128=/scratch/project_465002574/PDB/PDB_128
LATENT_20=/scratch/project_465002574/PDB/PDB_20
PDBBIND_LMDB=/scratch/project_465002574/PDB/PDBBind.lmdb
NCAA_LMDB=/scratch/project_465002574/PDB/NCAA/NCAA.lmdb

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# =============================================================================
# Build dataset flag arguments
# =============================================================================
DATASET_ARGS=""
[ "$USE_LMDB"    = "1" ] && DATASET_ARGS="$DATASET_ARGS --use-lmdb    --training-data $LMDB_DATA    --lmdb-weight $LMDB_WEIGHT"
[ "$USE_PDB"     = "1" ] && DATASET_ARGS="$DATASET_ARGS --use-pdb     --pdb-dir $PDB_DIR            --pdb-weight $PDB_WEIGHT --latent-root-128 $LATENT_128 --latent-root-20 $LATENT_20"
[ "$USE_PDBBIND" = "1" ] && DATASET_ARGS="$DATASET_ARGS --use-pdbbind --pdbbind-lmdb $PDBBIND_LMDB  --pdbbind-weight $PDBBIND_WEIGHT"
[ "$USE_NCAA"    = "1" ] && DATASET_ARGS="$DATASET_ARGS --use-ncaa    --ncaa-lmdb $NCAA_LMDB         --ncaa-weight $NCAA_WEIGHT"

# =============================================================================
# Rendezvous
# =============================================================================
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29522

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# =============================================================================
# Environment
# =============================================================================
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export PYTHONPATH="/flash/project_465002574/UAAG2_main/src:$PYTHONPATH"

export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
srun mkdir -p $MIOPEN_USER_DB_PATH

# =============================================================================
# Execution
# =============================================================================
srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export NODE_RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=0

    python -W ignore scripts/run_train.py \
      --gpus 1 \
      --num_nodes 8 \
      --batch-size 8 \
      --logger-type wandb \
      --id train_1node_test \
      --save-every-epoch \
      --use_protein_mpnn_context_128 \
      --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl \
      --num-workers 4 \
      --num-epochs 5 \
      --pdb-max-files 5000 \
      --pocket-dropout-prob 0.2 \
      --save-dir $OUTPUT_DIR \
      $DATASET_ARGS
"

srun rm -rf $MIOPEN_USER_DB_PATH
