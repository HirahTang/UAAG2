#!/bin/bash
#SBATCH --job-name=adaln_grover_cond
#SBATCH --account=project_465002988
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/project_465002574/uaag2_adit_results/logs/adaln_grover_cond_%A_%a.out
#SBATCH --error=/scratch/project_465002574/uaag2_adit_results/logs/adaln_grover_cond_%A_%a.err

# CP2 conditional-fidelity experiment.
# 40 conditions (20 natural + 20 CP2 benchmark NCAAs; DAL excluded as it is
# indistinguishable from ALA without stereochemistry) x 11 CP2 positions x 100 samples.
# 5 array elements x 8 GCDs = 40 slots, one condition per GCD.
#
# KNOWN CAVEAT: NCAA_PDB.lmdb (52% of this model's training by effective weight) has
# all-single bonds and no aromaticity, so NCAA condition latents seen in training were
# on a different chemistry from the correctly-drawn conditions used here. Run is
# deliberately exploratory.

module purge
module load LUMI CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default

export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export SRCDIR=/flash/project_465002574/UAAG2_conditional
export PYTHONPATH="${SRCDIR}/src:${PYTHONPATH:-}"
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0

export CKPT=${SRCDIR}/3DcoordsAtomsBonds_0/runConditional_adaLN_GROVER_cath1k_8gpu_20260904/epochepoch=199.ckpt
export BENCH_PT=/scratch/project_465002574/UNAAGI_benchmarks/5ly1_cp2.pt
export COND_TABLE=/scratch/project_465002574/condition_embeddings/cp2_experiment_conditions.npz
export SAMPLE_BASE=/scratch/project_465002574/CP2ConditionalFidelity_adaln_grover

mkdir -p "$SAMPLE_BASE" /scratch/project_465002574/uaag2_adit_results/logs

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

BASE=$(( SLURM_ARRAY_TASK_ID * 8 ))
echo "[$(date)] array ${SLURM_ARRAY_TASK_ID}: conditions ${BASE}..$(( BASE + 7 )) on $(hostname)"

srun --cpu-bind=mask_cpu:$MYMASKS bash -c "TASK_ID=\$(( ${BASE} + \$SLURM_PROCID )) ${SRCDIR}/slurm/cp2_cond_worker.sh"

echo "[$(date)] array ${SLURM_ARRAY_TASK_ID} complete"
