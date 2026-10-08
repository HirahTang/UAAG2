#!/bin/bash
#SBATCH --job-name=UAAG_v6
#SBATCH --account=project_465003401
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=47:59:00
#SBATCH -o /scratch/project_465003401/logs/train_v6_%x_%j.log
#SBATCH -e /scratch/project_465003401/logs/train_v6_%x_%j.log
#
# v6 pocket-conditional. Changes from v5.0, all from the UNAAGI-lite findings:
#   --all-atom-generation   CATH kept N/CA/C/O as fixed context while NCAA_indep
#                           generated whole molecules -> two tasks, one evaluation.
#   --virtual-node 0        node count was a sampler artefact, not a model property.
#   NCAA_PDB rebuilt        bond orders restored from the wwPDB CCD; the old LMDB
#                           stored every non-standard residue all-SINGLE.
#   +NCAA_BENCH weight 100  the only source of 13 benchmark NCAAs whose 3-letter
#                           codes collide with unrelated PDB components.
#   cond/pocket dropout 0.2 (was 0.5/0.5)
# occupy an extra atom type (9 vs 8) and an extra atom_mapping feature (26 vs 25), so
# loading it raises a size mismatch that strict=False does not tolerate. Removing virtual
# nodes and warm-starting from that prior are mutually exclusive; this trains from scratch.
set -u
MECH=additive
REP=${REP:?set REP=grover|morgan}
if [ "$REP" = grover ]; then
  TBL=/scratch/project_465002574/condition_embeddings/grover_base_condition_table.npz; NF=3201
else
  TBL=/scratch/project_465002574/condition_embeddings/morgan_r3_2048_condition_table.npz; NF=2049
fi
RUN_ID=v6_${MECH}_${REP}_$(date +%Y%m%d)

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 400))
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export WANDB_CACHE_DIR=/scratch/project_465003401/.wandb_cache
export WANDB_DATA_DIR=/scratch/project_465003401/.wandb_cache/data
export PYTHONPATH="/flash/project_465002574/UAAG2_conditional/src:${PYTHONPATH:-}"
export UAAG2_CCD_BONDS=/scratch/project_465003401/ccd/ccd_bond_orders.pkl
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
srun mkdir -p $MIOPEN_USER_DB_PATH
mkdir -p /scratch/project_465003401/logs

PRIOR_CKPT=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_8_gpu_UAAG_model_official_8_0203_prior/last-v1.ckpt
CATH1K=/scratch/project_465002574/cath_S35_res2.0_v4_3_0/structures_1k
NCAA_PDB_NEW=/scratch/project_465003401/ncaa_pdb_rebuild/final/NCAA_PDB.lmdb

cd /flash/project_465002574/UAAG2_conditional
srun python -W ignore scripts/run_train.py \
    --gpus 8 --num_nodes 1 \
    --batch-size 4 --accum-batch 4 --num-epochs 200 \
    --logger-type wandb --id ${RUN_ID} \
    --use-pdb --pdb-dir ${CATH1K} --pdb-weight 1.0 \
    --use-ncaa --ncaa-lmdb /scratch/project_465002574/PDB/NCAA/NCAA.lmdb --ncaa-weight 10.0 \
    --use-ncaa-pdb --ncaa-pdb-lmdb ${NCAA_PDB_NEW} --ncaa-pdb-weight 10.0 \
    --use-ncaa-bench --ncaa-bench-lmdb /scratch/project_465002574/PDB/NCAA_BENCH/NCAA.lmdb \
    --ncaa-bench-weight 100.0 \
    --all-atom-generation \
    --context-mapping --num-context-features ${NF} --cond-table ${TBL} \
    --cond-dropout 0.2 --cond-probe-batches 20 \
    --pocket-dropout-prob 0.2 \
    --virtual-node 0 --mask-rate 0.0 \
    --lr 0.0005 --grad-clip-val 1.0 --save-every-epoch --num-workers 4 \
    --save-dir /scratch/project_465003401/runs \
    --data_info_path /flash/project_465002574/UAAG2_conditional/data/statistic.pkl
