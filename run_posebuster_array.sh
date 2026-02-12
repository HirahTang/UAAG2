#!/bin/bash
#SBATCH --job-name=UAAG_posebuster
#SBATCH --account=project_465002574
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH -o /scratch/project_465002574/UAAG_logs/posebuster_%A_%a.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/posebuster_%A_%a.log

# ============================================================================
# UAAG PoseBusters Array Job
# ============================================================================
# Parallel PoseBusters evaluation: Each array task handles ONE iteration (0-1)
# 
# This script:
# 1. Extracts archived .mol files for the iteration
# 2. Runs PoseBusters evaluation
# 3. Cleans up temporary extracted files
# 
# Usage: Submitted automatically by submit_assay_jobs.sh
# ============================================================================

echo "============================================================================"
echo "UAAG PoseBusters Evaluation (Parallel)"
echo "============================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Iteration: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "============================================================================"

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_FILE=$1

if [ -z "${CONFIG_FILE}" ] || [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not provided or not found: ${CONFIG_FILE}"
    exit 1
fi

# Extract assay name from config file
ASSAY_NAME=$(basename ${CONFIG_FILE} .txt)
ITERATION=${SLURM_ARRAY_TASK_ID}

# Read first line to get common parameters
FIRST_LINE=$(awk 'NR==2 {print}' ${CONFIG_FILE})
PROTEIN_ID=$(echo ${FIRST_LINE} | awk '{print $2}')
MODEL="UAAG_model"
NUM_SAMPLES=1000

WORK_DIR=/flash/project_465002574/UAAG2_main
ARCHIVE_DIR=/scratch/project_465002574/UNAAGI_archives/${MODEL}
BASE_PATH=/scratch/project_465002574/ProteinGymSampling

echo "Assay: ${ASSAY_NAME}"
echo "Protein ID: ${PROTEIN_ID}"
echo "Model: ${MODEL}"
echo "Iteration: ${ITERATION}"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "→ Setting up environment..."

module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

cd ${WORK_DIR}

mkdir -p /scratch/project_465002574/UAAG_logs

# ============================================================================
# CHECK SAMPLES DIRECTORY
# ============================================================================
RUN_DIR="run${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}"
SAMPLES_DIR="${BASE_PATH}/${RUN_DIR}"

if [ ! -d "${SAMPLES_DIR}" ]; then
    echo "✗ ERROR: Samples directory not found - ${SAMPLES_DIR}"
    echo "  This job should run after sampling and post-processing complete"
    exit 1
fi

echo "✓ Samples directory found: ${SAMPLES_DIR}"

# Check if aa_distribution.csv exists (confirms post-processing completed)
if [ ! -f "${SAMPLES_DIR}/aa_distribution.csv" ]; then
    echo "⚠ WARNING: aa_distribution.csv not found"
    echo "  Post-processing may not have completed successfully"
fi

EXTRACTED_DIR="${SAMPLES_DIR}"

# ============================================================================
# STEP 1: RUN POSEBUSTERS EVALUATION
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 1: PoseBusters Evaluation (Iteration ${ITERATION})"
echo "============================================================================"
echo "[$(date)] Starting PoseBusters..."

OUTPUT_CSV="${SAMPLES_DIR}/PoseBusterResults"
TEMP_DIR="/flash/project_465002574/temp_sdf_posebuster_${SLURM_ARRAY_JOB_ID}_iter${ITERATION}"

echo "Input directory: ${SAMPLES_DIR}"
echo "Output: ${SAMPLES_DIR}"
echo "Temp directory: ${TEMP_DIR}"
echo ""

python scripts/evaluate_mol_samples.py \
    --input-dir "${SAMPLES_DIR}" \
    --output "${OUTPUT_CSV}" \
    --temp-dir "${TEMP_DIR}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ PoseBusters evaluation completed: ${OUTPUT_CSV}"
    
    # Check if summary file was created
    if [ -f "${OUTPUT_CSV}_summary.txt" ]; then
        echo "✓ Summary file created"
        cat "${OUTPUT_CSV}_summary.txt"
    fi
else
    echo ""
    echo "✗ PoseBusters evaluation failed for iteration ${ITERATION}"
    exit 1
fi


# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "Task ${SLURM_ARRAY_TASK_ID} Complete"
echo "============================================================================"
echo "Assay: ${ASSAY_NAME}"
echo "Protein: ${PROTEIN_ID}"
echo "Iteration: ${ITERATION}"
echo "Output: ${OUTPUT_CSV}"
echo "End time: $(date)"
echo "============================================================================"
