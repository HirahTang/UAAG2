#!/bin/bash
#SBATCH --job-name=UAAG_v4.2_mixin
#SBATCH --account=project_465002988
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH -o /flash/project_465002574/UAAG2_main/logs/train_flagship_v4.2_mixin_%j.log
#SBATCH -e /flash/project_465002574/UAAG2_main/logs/train_flagship_v4.2_mixin_%j.log

# Flagship v4.2 = continue the SOTA prior ckpt (0203_prior) on a MIX of the full
# unaagi_whole_v1.lmdb (weight 1) + NCAA_PDB.lmdb upweighted 10x. Contrast with
# v4.1, which continued on NCAA_PDB ONLY and regressed vs SOTA. The 10x is now a
# REAL WeightedRandomSampler (run_train.py patched 2026-07-17; old --weight flags
# were inert with sampler=None). Whole-LMDB ~millions of graphs, NCAA_PDB 38,863
# residues -> at 10x, NCAA_PDB is still a single-digit pct of each batch.
#
# NOTE vs v4.0: --num-epochs must exceed the resumed epoch (134). Lightning
# fit(ckpt_path=...) restores current_epoch/global_step, so max_epochs=50 would
# make the run exit immediately as "already finished". 634 = 134 + 500, so the
# 2-day walltime is the binding constraint, not the epoch counter.

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29523

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
# Redirect W&B cache to /scratch (huge quota) so it stops filling the 20G home
export WANDB_CACHE_DIR=/scratch/project_465002574/.wandb_cache
export WANDB_DATA_DIR=/scratch/project_465002574/.wandb_cache/data
export PYTHONPATH="/flash/project_465002574/UAAG2_main/src:${PYTHONPATH:-}"
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

RUN_ID=Flagship_v4.2_prior_mixin_ncaapdb10x_8gpu_20260717
PRIOR_CKPT=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_8_gpu_UAAG_model_official_8_0203_prior/last-v1.ckpt

srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export NODE_RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=0

    python -W ignore scripts/run_train.py \
        --gpus 1 \
        --num_nodes 8 \
        --load-ckpt ${PRIOR_CKPT} \
        --batch-size 4 \
        --accum-batch 4 \
        --num-epochs 634 \
        --logger-type wandb \
        --id ${RUN_ID} \
        --use-lmdb \
        --training-data /scratch/project_465002574/unaagi_whole_v1.lmdb \
        --lmdb-weight 1.0 \
        --use-ncaa-pdb \
        --ncaa-pdb-lmdb /scratch/project_465002574/PDB/NCAA_PDB_lmdb/NCAA_PDB.lmdb \
        --ncaa-pdb-weight 10.0 \
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
