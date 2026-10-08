#!/bin/bash
#SBATCH --job-name=morganfp_cfg
#SBATCH --account=project_465002988
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --array=0-24
#SBATCH --output=/scratch/project_465002574/uaag2_adit_results/logs/morganfp_cfg_%A_%a.out
#SBATCH --error=/scratch/project_465002574/uaag2_adit_results/logs/morganfp_cfg_%A_%a.err

# Classifier-free guidance sweep for UNAAGI-cond-MorganFP.
# 5 scales x 40 conditions = 200 tasks; 25 array elements x 8 GCDs.
# standard-g is EXCLUSIVE, so tasks are packed 8-per-node or we waste 8x.
# Uses the MORGAN condition table (2049-d) -- the GROVER one would be a dim mismatch.

module purge
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export SRCDIR=/flash/project_465002574/UAAG2_conditional
export PYTHONPATH="${SRCDIR}/src:${PYTHONPATH:-}"
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0

export CKPT=${SRCDIR}/3DcoordsAtomsBonds_0/runConditional_MorganFP_r3_2048_cath1k_8gpu_20260824/epochepoch=199.ckpt
export BENCH_PT=/scratch/project_465002574/UNAAGI_benchmarks/5ly1_cp2.pt
export COND_TABLE=/scratch/project_465002574/condition_embeddings/cp2_experiment_conditions_morgan.npz
export SAMPLE_BASE=/scratch/project_465002574/CP2_CFG_morganfp
mkdir -p "$SAMPLE_BASE" /scratch/project_465002574/uaag2_adit_results/logs

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"
BASE=$(( SLURM_ARRAY_TASK_ID * 8 ))
echo "[$(date)] array ${SLURM_ARRAY_TASK_ID}: global ${BASE}..$(( BASE + 7 )) on $(hostname)"

srun --cpu-bind=mask_cpu:$MYMASKS bash -c '
  SCALES=(1.0 1.5 2.0 3.0 5.0)
  GLOBAL=$(( '"$BASE"' + SLURM_PROCID ))
  SI=$(( GLOBAL / 40 ))
  export TASK_ID=$(( GLOBAL % 40 ))
  if [ "$SI" -ge 5 ]; then echo "[rank $SLURM_PROCID] global $GLOBAL -> no work"; exit 0; fi
  W=${SCALES[$SI]}
  export TAG="cfg_w${W}"
  export EXTRA_ARGS="--cfg-scale ${W}"
  echo "[rank $SLURM_PROCID] global=$GLOBAL w=$W cond_idx=$TASK_ID"
  bash '"$SRCDIR"'/slurm/cp2_cond_worker.sh
'
echo "[$(date)] array ${SLURM_ARRAY_TASK_ID} complete"
