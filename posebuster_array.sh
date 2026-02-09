#!/bin/bash
#SBATCH --job-name=posebuster_test
#SBATCH --account=project_465002574
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH -o logs/posebuster_%j.log
#SBATCH -e logs/posebuster_%j.log


echo "============================================================================"
echo "UAAG Array Job - Task ${SLURM_ARRAY_TASK_ID}"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================================"


# Switch to prior_condition branch
echo ""
echo "→ Switching to prior_condition branch..."
git checkout main

ASSAY_NAME=AICDA_HUMAN

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "→ Setting up environment..."

WORK_DIR=/flash/project_465002574/UAAG2_main
SAVE_DIR=/scratch/project_465002574/UNAAGI_archives/UAAG_model
# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
cd ${SAVE_DIR}
mkdir -p ${ASSAY_NAME}_iter${SLURM_ARRAY_TASK_ID}_extracted && tar -xzf /scratch/project_465002574/UNAAGI_archives/UAAG_model/${ASSAY_NAME}_UAAG_model_1000_iter${SLURM_ARRAY_TASK_ID}_mol_files.tar.gz -C ${ASSAY_NAME}_iter${SLURM_ARRAY_TASK_ID}_extracted
cd ${WORK_DIR}

python scripts/evaluate_mol_samples.py \
    --input-dir /scratch/project_465002574/UNAAGI_archives/UAAG_model/${ASSAY_NAME}_iter${SLURM_ARRAY_TASK_ID}_extracted \
    --output /scratch/project_465002574/ProteinGymSampling/runUAAG_model/${ASSAY_NAME}_UAAG_model_1000_iter${SLURM_ARRAY_TASK_ID}/PoseBusterResults \
    --temp-dir /flash/project_465002574/temp_sdf_iter${SLURM_ARRAY_TASK_ID}