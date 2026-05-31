#!/bin/bash
#SBATCH --job-name=UAAG_v0.3_ring_w8gpu_cont
#SBATCH --account=project_465002988
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH -o /flash/project_465002574/UAAG2_main/logs/train_flagship_v0.3_ring_weighted_cont_%j.log
#SBATCH -e /flash/project_465002574/UAAG2_main/logs/train_flagship_v0.3_ring_weighted_cont_%j.log

# ----------------------------------------------------------------------------
# Rendezvous
# ----------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29523

# ----------------------------------------------------------------------------
# CPU binding (Green Ring topology — 8 GCDs per LUMI-G node)
# ----------------------------------------------------------------------------
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export PYTHONPATH="/flash/project_465002574/UAAG2_main/src:${PYTHONPATH:-}"

# MI250X stability + NCCL on Slingshot
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

export WORK_DIR=/flash/project_465002574/UAAG2_main
cd ${WORK_DIR}
mkdir -p logs

srun mkdir -p $MIOPEN_USER_DB_PATH

RUN_ID=Flagship_v0.3_ring_weighted_8gpu_20260531_cont
PRIMARY_DIR=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFlagship_v0.3_ring_weighted_8gpu_20260531

# ----------------------------------------------------------------------------
# Execution — one task per GPU (8 tasks × 1 GPU each on 1 node)
# Lightning sees: --gpus 1 --num_nodes 8 (1 GPU per "node", 8 such "nodes")
# ----------------------------------------------------------------------------
srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export NODE_RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=0

    python -W ignore scripts/run_train.py \
        --gpus 1 \
        --num_nodes 8 \
        --batch-size 4 \
        --accum-batch 4 \
        --num-epochs 100 \
        --logger-type wandb \
        --id ${RUN_ID} \
        --load-ckpt ${PRIMARY_DIR}/last.ckpt \
        --use-pdb \
        --use-pdbbind \
        --use-ncaa \
        --pdb-dir /flash/project_465002574/UAAG2_main/data/pdb_subset_1000_v0.11_symlinks \
        --pdbbind-lmdb /scratch/project_465002574/PDB/PDBBind.lmdb \
        --ncaa-lmdb /scratch/project_465002574/PDB/NCAA/NCAA.lmdb \
        --pdb-weight 1.0 \
        --pdbbind-weight 10.0 \
        --ncaa-weight 10.0 \
        --pdb-max-files 1000 \
        --pocket-dropout-prob 0.5 \
        --mask-rate 0.0 \
        --max-virtual-nodes 11 \
        --lr 0.0005 \
        --grad-clip-val 1.0 \
        --save-every-epoch \
        --num-workers 4 \
        --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl
"

srun rm -rf $MIOPEN_USER_DB_PATH
