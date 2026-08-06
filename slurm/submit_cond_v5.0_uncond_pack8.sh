#!/bin/bash
#SBATCH --job-name=cond_uncond_pack8
#SBATCH --account=project_465002988
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --array=0-16
#SBATCH --output=/scratch/project_465002574/uaag2_adit_results/logs/cond_pack8_%A_%a.out
#SBATCH --error=/scratch/project_465002574/uaag2_adit_results/logs/cond_pack8_%A_%a.err

# Packed sampling: standard-g is OverSubscribe=EXCLUSIVE, so a 1-GPU request still
# bills the whole 8-GCD node. Instead of wasting 7/8 of it, run 8 independent
# (protein, iteration) units concurrently -- one per GCD -- via SLURM_PROCID.
#
# 27 proteins x 5 iters = 135 units; 17 array elements x 8 = 136 slots, the last
# one exits cleanly with no work. 17 nodes instead of 135.
#
# Note the tradeoff vs small-g: a node is held until its slowest of 8 finishes, so
# stragglers idle some GCDs. Still far better than 8x over-allocation, and
# standard-g usually schedules faster than small-g.

module purge
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default

export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export SRCDIR=/flash/project_465002574/UAAG2_conditional
export PYTHONPATH="${SRCDIR}/src:${PYTHONPATH:-}"
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0

export CKPT=${SRCDIR}/3DcoordsAtomsBonds_0/runConditional_v5.0_grover3201_cath1k_8gpu_20260804/epochepoch=199.ckpt
export MODEL_TAG=cond_v5.0_uncond
export BENCHMARK_DIR=/scratch/project_465002574/UNAAGI_benchmarks
export BASELINE_DIR=/scratch/project_465002574/UNAAGI_benchmark_values/baselines
export SAMPLE_BASE=/scratch/project_465002574/ProteinGymSampling
export RESULT_BASE=/scratch/project_465002574/UNAAGI_result/results/cond_v5.0_uncond
export NUM_SAMPLES=1000
export EXTRA_ARGS="--no-condition"     # unconditional branch of the conditional ckpt

mkdir -p /scratch/project_465002574/uaag2_adit_results/logs "$RESULT_BASE"

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

BASE=$(( SLURM_ARRAY_TASK_ID * 8 ))
echo "[$(date)] array ${SLURM_ARRAY_TASK_ID}: tasks ${BASE}..$(( BASE + 7 )) on $(hostname)"

srun --cpu-bind=mask_cpu:$MYMASKS bash -c "TASK_ID=\$(( ${BASE} + \$SLURM_PROCID )) ${SRCDIR}/slurm/pack8_worker.sh"

echo "[$(date)] array ${SLURM_ARRAY_TASK_ID} complete"
