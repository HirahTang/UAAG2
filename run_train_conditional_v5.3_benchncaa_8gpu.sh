#!/bin/bash
#SBATCH --job-name=UAAG_cond_v5.3
#SBATCH --account=project_465002988
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH -o /flash/project_465002574/UAAG2_conditional/logs/train_cond_v5.3_%j.log
#SBATCH -e /flash/project_465002574/UAAG2_conditional/logs/train_cond_v5.3_%j.log

# Conditional UAAG2 v5.0 — static GROVER topology condition on the sidechain nodes.
#
# Condition: frozen GROVER-base embedding (3201-d, mean||max||n_atoms) of the WHOLE
# residue topology (is_ligand==1, backbone N/CA/C/O included), looked up from a static
# table. Injected only into the nodes being denoised (is_ligand - is_backbone).
# Dropped per-graph 50% of the time -> the unconditional branch stays exactly the prior's,
# which is what classifier-free guidance needs later.
#
# Warm start: WEIGHTS ONLY from the p0203 SOTA prior (--load-prior-weights). Lightning's
# fit(ckpt_path=) is a strict load and rejects the prior now that the context layers
# exist. atom_context_mapping is zero-initialised, so step 0 == the prior exactly.
#
# Data (per user, 2026-08-04): CATH S35 res<=2.0 selected 1k for the natural-PDB part,
# plus independent NCAA + NCAA_PDB + PDBBind. NOT unaagi_whole_v1.
# Hyperparameters otherwise identical to v4.2: 11 virtual nodes, 0.5 pocket dropout.

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29551

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export WANDB_CACHE_DIR=/scratch/project_465002574/.wandb_cache
export WANDB_DATA_DIR=/scratch/project_465002574/.wandb_cache/data
export PYTHONPATH="/flash/project_465002574/UAAG2_conditional/src:${PYTHONPATH:-}"
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

export WORK_DIR=/flash/project_465002574/UAAG2_conditional
cd ${WORK_DIR}
mkdir -p logs
srun mkdir -p $MIOPEN_USER_DB_PATH

RUN_ID=Conditional_v5.3_benchncaa_cont_8gpu_20260821
CONT_CKPT=${WORK_DIR}/3DcoordsAtomsBonds_0/runConditional_v5.2_cathfull_cont_8gpu_20260817/epochepoch=382.ckpt
COND_TABLE=/scratch/project_465002574/condition_embeddings/grover_base_condition_table.npz
CATH1K=/scratch/project_465002574/cath_S35_res2.0_v4_3_0/structures

srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export NODE_RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=0

    python -W ignore scripts/run_train.py \
        --gpus 1 \
        --num_nodes 8 \
        --load-ckpt ${CONT_CKPT} \
        --batch-size 4 \
        --accum-batch 4 \
        --num-epochs 600 \
        --logger-type wandb \
        --id ${RUN_ID} \
        --use-pdb \
        --pdb-dir ${CATH1K} \
        --pdb-weight 1.0 \
        --use-ncaa \
        --ncaa-lmdb /scratch/project_465002574/PDB/NCAA/NCAA.lmdb \
        --ncaa-weight 10.0 \
        --use-ncaa-pdb \
        --ncaa-pdb-lmdb /scratch/project_465002574/PDB/NCAA_PDB_lmdb/NCAA_PDB.lmdb \
        --ncaa-pdb-weight 10.0 \
        --use-ncaa-bench \
        --ncaa-bench-lmdb /scratch/project_465002574/PDB/NCAA_BENCH/NCAA.lmdb \
        --ncaa-bench-weight 50.0 \
        --use-pdbbind \
        --pdbbind-lmdb /scratch/project_465002574/PDB/PDBBind.lmdb \
        --pdbbind-weight 10.0 \
        --context-mapping \
        --num-context-features 3201 \
        --cond-table ${COND_TABLE} \
        --cond-dropout 0.5 \
        --cond-probe-batches 20 \
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
