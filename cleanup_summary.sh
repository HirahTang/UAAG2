#!/bin/bash
# ============================================================================
# UAAG Cleanup Summary Script
# ============================================================================
# Run this AFTER the array job completes to see overall results
# Usage: bash cleanup_summary.sh <protein_id> <model_name> <num_samples>
# ============================================================================

PROTEIN_ID=${1:-A0A247D711_LISMN}
MODEL=${2:-UAAG_model}
NUM_SAMPLES=${3:-1000}

BASE_PATH=/scratch/project_465002574/ProteinGymSampling
ARCHIVE_DIR=/scratch/project_465002574/UNAAGI_archives/${MODEL}

echo "============================================================================"
echo "UAAG Cleanup Summary"
echo "============================================================================"
echo "Protein: ${PROTEIN_ID}"
echo "Model: ${MODEL}"
echo "Samples: ${NUM_SAMPLES}"
echo "Date: $(date)"
echo ""

echo "============================================================================"
echo "Archives Created"
echo "============================================================================"
ls -lh ${ARCHIVE_DIR}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter*_mol_files.tar.gz 2>/dev/null

TOTAL_ARCHIVE_SIZE=$(du -sh ${ARCHIVE_DIR}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter*_mol_files.tar.gz 2>/dev/null | awk '{sum+=$1} END {print sum}')
echo ""
echo "Total archive size: ${TOTAL_ARCHIVE_SIZE}"

echo ""
echo "============================================================================"
echo "Iteration Status"
echo "============================================================================"

TOTAL_MOL_FILES=0
TOTAL_DISK_USAGE=0

for ITERATION in {0..4}; do
    RUN_DIR="run${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}"
    SAMPLES_DIR="${BASE_PATH}/${RUN_DIR}/Samples"
    
    if [ ! -d "${SAMPLES_DIR}" ]; then
        echo "Iteration ${ITERATION}: ✗ Directory not found"
        continue
    fi
    
    # Count remaining .mol files
    REMAINING_MOL=$(find "${SAMPLES_DIR}" -name "*.mol" 2>/dev/null | wc -l)
    TOTAL_MOL_FILES=$((TOTAL_MOL_FILES + REMAINING_MOL))
    
    # Check important files
    AA_DIST=$([ -f "${SAMPLES_DIR}/aa_distribution.csv" ] && echo "✓" || echo "✗")
    POSEBUST=$([ -f "${SAMPLES_DIR}/posebusters_evaluation.csv" ] && echo "✓" || echo "✗")
    
    # Disk usage
    DISK_USAGE=$(du -sh "${SAMPLES_DIR}" 2>/dev/null | cut -f1)
    
    echo ""
    echo "Iteration ${ITERATION}:"
    echo "  Directory: ${SAMPLES_DIR}"
    echo "  Remaining .mol files: ${REMAINING_MOL}"
    echo "  aa_distribution.csv: ${AA_DIST}"
    echo "  posebusters_evaluation.csv: ${POSEBUST}"
    echo "  Disk usage: ${DISK_USAGE}"
done

echo ""
echo "============================================================================"
echo "Overall Statistics"
echo "============================================================================"
echo "Total remaining .mol files across all iterations: ${TOTAL_MOL_FILES}"
echo ""

if [ ${TOTAL_MOL_FILES} -eq 0 ]; then
    echo "✓ SUCCESS: All .mol files cleaned up"
else
    echo "⚠ WARNING: ${TOTAL_MOL_FILES} .mol files still present"
fi

echo ""
echo "============================================================================"
