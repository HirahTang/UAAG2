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
    echo "  Pipeline: Sampling → Post-analysis → PoseBusters → Compress → Delete"
    echo ""
    
    # Step 1: Submit the sampling array job (50 tasks)
    SAMPLING_JOB=$(sbatch --parsable \
                          --job-name="UAAG_samp_${assay_name}" \
                          --export=ALL,CONFIG_FILE=${config_file} \
                          ${SCRIPT} ${config_file} 2>&1)
    
    if [[ ${SAMPLING_JOB} =~ ^[0-9]+$ ]]; then
        echo "  ✓ Step 1: Sampling submitted: ${SAMPLING_JOB} (50 array tasks: 5 iter × 10 splits)"
        
        # Step 2: Submit post-processing job (generates aa_distribution.csv)
        POSTPROC_JOB=$(sbatch --parsable \
                              --job-name="UAAG_post_${assay_name}" \
                              --dependency=afterok:${SAMPLING_JOB} \
                              run_assay_postprocess.sh ${config_file} 2>&1)
        
        if [[ ${POSTPROC_JOB} =~ ^[0-9]+$ ]]; then
            echo "  ✓ Step 2: Post-analysis submitted: ${POSTPROC_JOB} (generates aa_distribution.csv)"
            
            # Step 3: Submit PoseBusters array job (5 tasks, one per iteration)
            POSEBUSTER_JOB=$(sbatch --parsable \
                                    --job-name="UAAG_pb_${assay_name}" \
                                    --dependency=afterok:${POSTPROC_JOB} \
                                    run_posebuster_array.sh ${config_file} 2>&1)
            
            if [[ ${POSEBUSTER_JOB} =~ ^[0-9]+$ ]]; then
                echo "  ✓ Step 3: PoseBusters submitted: ${POSEBUSTER_JOB} (5 array tasks: evaluates .mol files)"
                
                # Step 4: Submit cleanup array job (compress & delete)
                CLEANUP_JOB=$(sbatch --parsable \
                                     --job-name="UAAG_clean_${assay_name}" \
                                     --dependency=afterok:${POSEBUSTER_JOB} \
                                     cleanup_mol_files_array.sh ${assay_name} UAAG_model 1000 2>&1)
                
                if [[ ${CLEANUP_JOB} =~ ^[0-9]+$ ]]; then
                    echo "  ✓ Step 4: Cleanup submitted: ${CLEANUP_JOB} (5 array tasks: compress & delete)"
                else
                    echo "  ✗ Cleanup submission failed: ${CLEANUP_JOB}"
                fi
            else
                echo "  ✗ PoseBusters submission failed: ${POSEBUSTER_JOB}"
            fi
        else
            echo "  ✗ Post-analysis submission failed: ${POSTPROC_JOB}"
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
