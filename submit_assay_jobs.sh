#!/bin/bash

# ============================================================================
# UAAG Array Job Submission Script
# ============================================================================
# Submits array jobs for UAAG assays
# Usage:
#   ./submit_assay_jobs.sh <assay_name>        # Submit one assay
#   ./submit_assay_jobs.sh all                  # Submit all 25 assays
# ============================================================================

ASSAY_DIR="slurm_config/assays"
SCRIPT="run_assay_array.sh"

if [ ! -f "${SCRIPT}" ]; then
    echo "Error: ${SCRIPT} not found!"
    exit 1
fi

submit_assay() {
    local config_file=$1
    local assay_name=$(basename ${config_file} .txt)
    
    echo "Submitting jobs for: ${assay_name}"
    echo "  Config: ${config_file}"
    echo "  Sampling tasks: 0-49 (5 iterations × 10 splits)"
    
    # Submit the sampling array job
    SAMPLING_JOB=$(sbatch --parsable \
                          --job-name="UAAG_samp_${assay_name}" \
                          --export=ALL,CONFIG_FILE=${config_file} \
                          ${SCRIPT} ${config_file} 2>&1)
    
    if [[ ${SAMPLING_JOB} =~ ^[0-9]+$ ]]; then
        echo "  ✓ Sampling job submitted: ${SAMPLING_JOB} (50 array tasks)"
        
        # Submit post-processing job with dependency on sampling completion
        POSTPROC_JOB=$(sbatch --parsable \
                              --job-name="UAAG_post_${assay_name}" \
                              --dependency=afterok:${SAMPLING_JOB} \
                              run_assay_postprocess.sh ${config_file} 2>&1)
        
        if [[ ${POSTPROC_JOB} =~ ^[0-9]+$ ]]; then
            echo "  ✓ Post-processing job submitted: ${POSTPROC_JOB} (runs after ${SAMPLING_JOB})"
        else
            echo "  ✗ Post-processing submission failed: ${POSTPROC_JOB}"
        fi
    else
        echo "  ✗ Sampling job submission failed: ${SAMPLING_JOB}"
    fi
    echo ""
}

# ============================================================================
# MAIN
# ============================================================================

if [ $# -eq 0 ]; then
    echo "Usage: $0 <assay_name|all>"
    echo ""
    echo "Available assays:"
    ls -1 ${ASSAY_DIR}/*.txt | xargs -n1 basename | sed 's/.txt$//' | nl
    echo ""
    echo "Examples:"
    echo "  $0 ENVZ_ECOLI       # Submit ENVZ_ECOLI only"
    echo "  $0 DN7A_SACS2       # Submit DN7A_SACS2 only"
    echo "  $0 all              # Submit all 25 assays"
    exit 0
fi

ASSAY_NAME=$1

if [ "${ASSAY_NAME}" == "all" ]; then
    echo "============================================================================"
    echo "Submitting ALL 25 assays (1,250 total tasks)"
    echo "============================================================================"
    echo ""
    
    for config_file in ${ASSAY_DIR}/*.txt; do
        submit_assay ${config_file}
        sleep 1  # Brief delay between submissions
    done
    
    echo "============================================================================"
    echo "Submission complete!"
    echo "Check job status with: squeue -u \$USER"
    echo "============================================================================"
    
elif [ -f "${ASSAY_DIR}/${ASSAY_NAME}.txt" ]; then
    echo "============================================================================"
    echo "Submitting single assay: ${ASSAY_NAME}"
    echo "============================================================================"
    echo ""
    
    submit_assay "${ASSAY_DIR}/${ASSAY_NAME}.txt"
    
    echo "============================================================================"
    echo "Check job status with: squeue -u \$USER"
    echo "============================================================================"
    
else
    echo "Error: Assay '${ASSAY_NAME}' not found!"
    echo ""
    echo "Available assays:"
    ls -1 ${ASSAY_DIR}/*.txt | xargs -n1 basename | sed 's/.txt$//' | nl
    exit 1
fi
