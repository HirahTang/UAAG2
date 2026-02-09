#!/bin/bash
#SBATCH --job-name=posebuster_test
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH -o logs/posebuster_%j.log
#SBATCH -e logs/posebuster_%j.log


# Switch to prior_condition branch
echo ""
echo "→ Switching to prior_condition branch..."
git checkout main


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "→ Setting up environment..."

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

mkdir -p /scratch/project_465002574/UAAG_logs

MODEL="UAAG_model"
NUM_SAMPLES=1000
PROTEIN_ID="ARGR_ECOLI"
# ============================================================================
# CHECK SAMPLES DIRECTORY
# ============================================================================
BASE_PATH=/scratch/project_465002574/ProteinGymSampling
RUN_DIR="run${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${SLURM_ARRAY_TASK_ID}"
SAMPLES_DIR="${BASE_PATH}/${RUN_DIR}"


WORK_DIR=/flash/project_465002574/UAAG2_main

SAVE_DIR=/scratch/project_465002574/UNAAGI_archives/UAAG_model
OUTPUT_CSV="${SAMPLES_DIR}/PoseBusterResults"
TEMP_DIR="/flash/project_465002574/temp_sdf_posebuster_${SLURM_ARRAY_JOB_ID}_iter${SLURM_ARRAY_TASK_ID}"


python scripts/evaluate_mol_samples.py \
    --input-dir "${SAMPLES_DIR}" \
    --output "${OUTPUT_CSV}" \
    --temp-dir "${TEMP_DIR}"
