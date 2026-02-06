#!/bin/bash
#SBATCH --job-name=UAAG_cleanup
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=5:00:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/cleanup_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/cleanup_%j.log

# ============================================================================
# UAAG Cleanup Script
# ============================================================================
# 1. Evaluates .mol files with PoseBusters
# 2. Compresses all .mol files into archives
# 3. Deletes .mol files while keeping CSV and JSON results
# 
# Usage: sbatch cleanup_mol_files.sh <protein_id> <model_name> <num_samples>
# Example: sbatch cleanup_mol_files.sh A0A247D711_LISMN UAAG_model 1000
# ============================================================================

echo "============================================================================"
echo "UAAG Cleanup Script"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "============================================================================"

# ============================================================================
# CONFIGURATION
# ============================================================================
PROTEIN_ID=${1:-A0A247D711_LISMN}
MODEL=${2:-UAAG_model}
NUM_SAMPLES=${3:-1000}

WORK_DIR=/flash/project_465002574/UAAG2_main
BASE_PATH=/scratch/project_465002574/ProteinGymSampling
ARCHIVE_DIR=/scratch/project_465002574/UNAAGI_archives/${MODEL}

echo "Protein ID: ${PROTEIN_ID}"
echo "Model: ${MODEL}"
echo "Samples: ${NUM_SAMPLES}"
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

mkdir -p ${ARCHIVE_DIR}
mkdir -p /scratch/project_465002574/UAAG_logs

# ============================================================================
# STEP 1: EVALUATE WITH POSEBUSTERS
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 1: PoseBusters Evaluation"
echo "============================================================================"

for ITERATION in {0..4}; do
    RUN_DIR="run${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}"
    SAMPLES_DIR="${BASE_PATH}/${RUN_DIR}/Samples"
    
    if [ ! -d "${SAMPLES_DIR}" ]; then
        echo "⚠ Skipping iteration ${ITERATION}: Directory not found - ${SAMPLES_DIR}"
        continue
    fi
    
    echo ""
    echo "[$(date)] Evaluating iteration ${ITERATION}..."
    echo "  Directory: ${SAMPLES_DIR}"
    
    # Run PoseBusters evaluation
    OUTPUT_CSV="${SAMPLES_DIR}/posebusters_evaluation.csv"
    
    python scripts/evaluate_mol_samples.py \
        --input-dir "${SAMPLES_DIR}" \
        -o "${OUTPUT_CSV}" \
        --max-workers 6 \
        --temp-dir "/scratch/project_465002574/temp_sdf_cleanup_iter${ITERATION}"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Evaluation completed: ${OUTPUT_CSV}"
    else
        echo "  ✗ Evaluation failed for iteration ${ITERATION}"
    fi
done

# ============================================================================
# STEP 2: COMPRESS .mol FILES
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 2: Compressing .mol Files"
echo "============================================================================"

for ITERATION in {0..4}; do
    RUN_DIR="run${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}"
    SAMPLES_DIR="${BASE_PATH}/${RUN_DIR}/Samples"
    
    if [ ! -d "${SAMPLES_DIR}" ]; then
        continue
    fi
    
    echo ""
    echo "[$(date)] Compressing iteration ${ITERATION}..."
    
    # Create archive with all .mol files
    ARCHIVE_NAME="${ARCHIVE_DIR}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}_mol_files.tar.gz"
    
    # Find and compress all .mol files (including all_molecules.sdf if using consolidated output)
    cd "${SAMPLES_DIR}"
    
    # Count .mol files
    MOL_COUNT=$(find . -name "*.mol" -o -name "all_molecules.sdf" | wc -l)
    
    if [ ${MOL_COUNT} -gt 0 ]; then
        echo "  Found ${MOL_COUNT} molecule files"
        
        # Create tar archive with .mol and .sdf files
        find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -print0 | \
            tar -czf "${ARCHIVE_NAME}" --null -T -
        
        if [ $? -eq 0 ]; then
            ARCHIVE_SIZE=$(du -h "${ARCHIVE_NAME}" | cut -f1)
            echo "  ✓ Archive created: ${ARCHIVE_NAME} (${ARCHIVE_SIZE})"
        else
            echo "  ✗ Failed to create archive for iteration ${ITERATION}"
            continue
        fi
    else
        echo "  ⚠ No molecule files found in iteration ${ITERATION}"
    fi
done

cd ${WORK_DIR}

# ============================================================================
# STEP 3: DELETE .mol FILES (KEEP CSV/JSON)
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 3: Cleaning Up .mol Files"
echo "============================================================================"
echo ""
echo "Files to DELETE:"
echo "  - *.mol files"
echo "  - all_molecules.sdf files"
echo "  - Temporary directories (batch_*, iter_*)"
echo ""
echo "Files to KEEP:"
echo "  - *_aa_table.csv"
echo "  - results.json"
echo "  - aa_distribution.csv"
echo "  - posebusters_evaluation.csv"
echo "  - posebusters_evaluation_summary.txt"
echo ""

for ITERATION in {0..4}; do
    RUN_DIR="run${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}"
    SAMPLES_DIR="${BASE_PATH}/${RUN_DIR}/Samples"
    
    if [ ! -d "${SAMPLES_DIR}" ]; then
        continue
    fi
    
    echo "[$(date)] Cleaning iteration ${ITERATION}..."
    cd "${SAMPLES_DIR}"
    
    # Count files before deletion
    MOL_COUNT=$(find . -name "*.mol" | wc -l)
    SDF_COUNT=$(find . -name "all_molecules.sdf" | wc -l)
    TOTAL_TO_DELETE=$((MOL_COUNT + SDF_COUNT))
    
    echo "  Deleting ${TOTAL_TO_DELETE} molecule files (${MOL_COUNT} .mol + ${SDF_COUNT} .sdf)..."
    
    # Delete all .mol files
    find . -name "*.mol" -type f -delete
    
    # Delete all_molecules.sdf files
    find . -name "all_molecules.sdf" -type f -delete
    
    # Delete batch_* and iter_* directories (they should be empty now or contain only intermediate files)
    find . -type d -name "batch_*" -exec rm -rf {} + 2>/dev/null
    find . -type d -name "iter_*" -exec rm -rf {} + 2>/dev/null
    find . -type d -name "final" -exec rm -rf {} + 2>/dev/null
    
    # Verify important files are still there
    CSV_COUNT=$(find . -name "*_aa_table.csv" -o -name "aa_distribution.csv" -o -name "posebusters_evaluation.csv" | wc -l)
    JSON_COUNT=$(find . -name "results.json" | wc -l)
    
    echo "  ✓ Cleanup complete"
    echo "    Remaining important files: ${CSV_COUNT} CSV files, ${JSON_COUNT} JSON files"
done

cd ${WORK_DIR}

# ============================================================================
# STEP 4: VERIFICATION
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 4: Verification"
echo "============================================================================"

for ITERATION in {0..4}; do
    RUN_DIR="run${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}"
    SAMPLES_DIR="${BASE_PATH}/${RUN_DIR}/Samples"
    
    if [ ! -d "${SAMPLES_DIR}" ]; then
        continue
    fi
    
    echo ""
    echo "Iteration ${ITERATION}:"
    echo "  Directory: ${SAMPLES_DIR}"
    
    # Check for remaining .mol files (should be 0)
    REMAINING_MOL=$(find "${SAMPLES_DIR}" -name "*.mol" | wc -l)
    
    # Check for important files
    AA_DIST=$([ -f "${SAMPLES_DIR}/aa_distribution.csv" ] && echo "✓" || echo "✗")
    POSEBUST=$([ -f "${SAMPLES_DIR}/posebusters_evaluation.csv" ] && echo "✓" || echo "✗")
    
    echo "  Remaining .mol files: ${REMAINING_MOL} (should be 0)"
    echo "  aa_distribution.csv: ${AA_DIST}"
    echo "  posebusters_evaluation.csv: ${POSEBUST}"
    
    # Disk usage
    DISK_USAGE=$(du -sh "${SAMPLES_DIR}" | cut -f1)
    echo "  Disk usage: ${DISK_USAGE}"
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "Cleanup Summary"
echo "============================================================================"
echo "Protein: ${PROTEIN_ID}"
echo "Model: ${MODEL}"
echo "Iterations processed: 0-4"
echo ""
echo "Archives created in: ${ARCHIVE_DIR}"
ls -lh ${ARCHIVE_DIR}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter*_mol_files.tar.gz 2>/dev/null || echo "  (no archives found)"
echo ""
echo "End time: $(date)"
echo "============================================================================"
