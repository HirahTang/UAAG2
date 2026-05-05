#!/bin/bash
#SBATCH --job-name=prior_pb_cleanup
#SBATCH --account=project_465002574
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/prior_cleanup_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/prior_cleanup_%j.log

# ============================================================================
# Prior Sampling Post-Run PoseBusters + Archive + Cleanup
# ============================================================================
# Run this AFTER generate_ligand.py, post_analysis.py, and result_eval_uniform_uaa.py
# are already complete.
#
# Usage examples:
#   sbatch run_prior_posebuster_cleanup.sh PRIOR_FREE_UAAG_model_1000_split0
#   sbatch run_prior_posebuster_cleanup.sh UAAG_model/PRIOR_FREE_UAAG_model_1000_split0
#
# Optional args:
#   $2 = model name (default: UAAG_model)
#   $3 = samples subfolder (default: Samples_prior)
# ============================================================================

echo "============================================================================"
echo "Prior PoseBusters + Cleanup"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "============================================================================"

RUN_FOLDER_INPUT=$1
MODEL=${2:-UAAG_model}
SAMPLES_SUBDIR=${3:-Samples_prior}

if [ -z "${RUN_FOLDER_INPUT}" ]; then
    echo "Error: run folder name is required"
    echo "Usage: sbatch $0 <run_folder> [model] [samples_subdir]"
    echo "Example: sbatch $0 PRIOR_FREE_UAAG_model_1000_split0"
    exit 1
fi

WORK_DIR=/flash/project_465002574/UAAG2_main
BASE_PATH=/scratch/project_465002574/ProteinGymSampling
ARCHIVE_DIR=/scratch/project_465002574/UNAAGI_archives/${MODEL}

# Support both inputs:
# 1) PRIOR_FREE_UAAG_model_1000_split0
# 2) UAAG_model/PRIOR_FREE_UAAG_model_1000_split0
if [[ "${RUN_FOLDER_INPUT}" == */* ]]; then
    RUN_REL_PATH="${RUN_FOLDER_INPUT}"
else
    RUN_REL_PATH="${MODEL}/${RUN_FOLDER_INPUT}"
fi

RUN_BASE_DIR="${BASE_PATH}/run${RUN_REL_PATH}"
SAMPLES_PATH="${RUN_BASE_DIR}/${SAMPLES_SUBDIR}"
POSEBUSTER_OUTPUT="${SAMPLES_PATH}/PoseBusterResults"
POSEBUSTER_TEMP_DIR="/flash/project_465002574/temp_sdf_${SLURM_JOB_ID}"
ARCHIVE_NAME="$(echo "${RUN_REL_PATH}" | tr '/' '_')_mol_files.tar.gz"
ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}"

echo "Run input: ${RUN_FOLDER_INPUT}"
echo "Resolved run path: run${RUN_REL_PATH}"
echo "Samples path: ${SAMPLES_PATH}"
echo "Archive path: ${ARCHIVE_PATH}"
echo ""

if [ ! -d "${SAMPLES_PATH}" ]; then
    echo "Error: samples directory not found: ${SAMPLES_PATH}"
    exit 1
fi

# Environment setup
echo "→ Setting up environment..."
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

mkdir -p ${ARCHIVE_DIR}
cd ${WORK_DIR}

# ============================================================================
# STEP 1: PoseBusters Evaluation
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 1: PoseBusters Evaluation"
echo "============================================================================"
echo "[$(date)] Evaluating molecules in ${SAMPLES_PATH}..."

python scripts/evaluate_mol_samples.py \
    --input-dir ${SAMPLES_PATH} \
    --output ${POSEBUSTER_OUTPUT} \
    --max-workers 6 \
    --temp-dir ${POSEBUSTER_TEMP_DIR}

if [ $? -ne 0 ]; then
    echo "[$(date)] PoseBusters evaluation failed"
    exit 1
fi

if [ -f "${POSEBUSTER_OUTPUT}" ]; then
    EVAL_LINES=$(wc -l < "${POSEBUSTER_OUTPUT}")
    echo "[$(date)] PoseBusters completed: $((EVAL_LINES - 1)) molecules evaluated"
else
    echo "[$(date)] Warning: PoseBusters command succeeded but output file not found"
fi

# ============================================================================
# STEP 2: Compress molecule files
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 2: Compress Molecule Files"
echo "============================================================================"

cd ${SAMPLES_PATH}
MOL_COUNT=$(find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -type f | wc -l)

echo "Found ${MOL_COUNT} molecule files to archive"

if [ ${MOL_COUNT} -eq 0 ]; then
    echo "No molecule files found. Skipping archive and cleanup."
    exit 0
fi

TAR_EXIT_CODE=0
if tar --help 2>/dev/null | grep -q -- "--checkpoint-action"; then
    find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -type f -print0 | \
        tar --null -T - -czf "${ARCHIVE_PATH}" \
            --checkpoint=2000 \
            --checkpoint-action=echo='[tar] checkpoints processed: %u'
    TAR_EXIT_CODE=$?
else
    find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -type f -print0 | \
        tar -czf "${ARCHIVE_PATH}" --null -T -
    TAR_EXIT_CODE=$?
fi

if [ ${TAR_EXIT_CODE} -ne 0 ]; then
    echo "[$(date)] Archive creation failed. Skipping cleanup for safety."
    exit 1
fi

ARCHIVE_SIZE=$(du -h "${ARCHIVE_PATH}" | cut -f1)
echo "[$(date)] Archive created: ${ARCHIVE_PATH} (${ARCHIVE_SIZE})"

# ============================================================================
# STEP 3: Cleanup raw molecule files
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 3: Cleanup Raw Molecule Files"
echo "============================================================================"

MOL_BEFORE=$(find . -name "*.mol" -type f | wc -l)
SDF_BEFORE=$(find . -name "all_molecules.sdf" -type f | wc -l)
echo "Deleting $((MOL_BEFORE + SDF_BEFORE)) molecule files (${MOL_BEFORE} .mol + ${SDF_BEFORE} .sdf)"

find . -name "*.mol" -type f -delete
find . -name "all_molecules.sdf" -type f -delete
find . -type d -name "batch_*" -exec rm -rf {} + 2>/dev/null
find . -type d -name "iter_*" -exec rm -rf {} + 2>/dev/null
find . -type d -name "final" -exec rm -rf {} + 2>/dev/null

REMAINING_MOL=$(find . -name "*.mol" -type f | wc -l)
REMAINING_SDF=$(find . -name "all_molecules.sdf" -type f | wc -l)
DISK_USAGE=$(du -sh ${SAMPLES_PATH} | cut -f1)

echo "Cleanup complete"
echo "Remaining molecule files: $((REMAINING_MOL + REMAINING_SDF))"
echo "Samples directory size: ${DISK_USAGE}"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "Prior PoseBusters + Cleanup Completed"
echo "============================================================================"
echo "End time: $(date)"
echo "Samples path: ${SAMPLES_PATH}"
echo "PoseBusters output: ${POSEBUSTER_OUTPUT}"
echo "Archive path: ${ARCHIVE_PATH}"
echo "============================================================================"
