#!/bin/bash
#SBATCH --job-name=UAAG_adaln_MorganFP
#SBATCH --account=project_465002988
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH -o /flash/project_465002574/UAAG2_conditional/logs/train_adaln_morgan_%j.log
#SBATCH -e /flash/project_465002574/UAAG2_conditional/logs/train_adaln_morgan_%j.log
# UNAAGI-cond-MorganFP — rule-based Morgan fingerprint condition on the sidechain nodes.
#
# Condition: count-based Morgan fingerprint (radius=3, 2048 bits) of the WHOLE residue
# topology (is_ligand==1, backbone N/CA/C/O included), concatenated with the heavy-atom
# count = 2049-d, looked up from a static table. Injected only into the nodes being
# denoised (is_ligand - is_backbone). Dropped per-graph 50% of the time, so the
# unconditional branch stays exactly the prior's, which is what CFG needs later.
#
# WHY: the v5.0 alignment analysis showed the GROVER latent CANNOT name a single residue —
# homologs collapse (ASP/GLU 0.974, ILE/LEU 0.995 against a 0.380 background) and no amount
# of training reaches them. Morgan is rule-based and resolves them: on the 40 CP2 eval
# conditions the background falls to 0.052 (p90 0.116) with ZERO pairs above 0.95, worst
# pair CHA/CPA 0.896 (was 0.996). This run tests whether that separability converts into
# controllability.
#
# STRICT A/B vs v5.0: the Morgan table reuses v5.0's EXACT molecule rows, record mapping and
# source labels — only the vectors differ. Every other hyperparameter matches v5.0
# (CATH-1k, 200 epochs, 11 virtual nodes, 0.5 pocket dropout, lr 5e-4, clip 1.0), so any
# difference is attributable to the condition representation alone. Deliberately NOT
# CATH-full: v5.0 -> v5.2 showed that data change alone moves CP2 by +0.100.
#
# Warm start: WEIGHTS ONLY from the p0203 SOTA prior (--load-prior-weights).
# atom_context_mapping is zero-initialised, so step 0 == the prior exactly.

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29531

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

RUN_ID=Conditional_adaLN_MorganFP_cath1k_8gpu_20260904
PRIOR_CKPT=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_8_gpu_UAAG_model_official_8_0203_prior/last-v1.ckpt
COND_TABLE=/scratch/project_465002574/condition_embeddings/morgan_r3_2048_condition_table.npz
CATH1K=/scratch/project_465002574/cath_S35_res2.0_v4_3_0/structures_1k

srun --cpu-bind=mask_cpu:$MYMASKS bash -c "
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export NODE_RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=0

    python -W ignore scripts/run_train.py \
        --gpus 1 \
        --num_nodes 8 \
        --load-prior-weights ${PRIOR_CKPT} \
        --batch-size 4 \
        --accum-batch 4 \
        --num-epochs 200 \
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
        --use-pdbbind \
        --pdbbind-lmdb /scratch/project_465002574/PDB/PDBBind.lmdb \
        --pdbbind-weight 10.0 \
        --context-mapping \
        --adaln-context \
        --adaln-scope graph \
        --num-context-features 2049 \
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
