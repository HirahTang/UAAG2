#!/bin/bash
#SBATCH --job-name=UAAG_postproc
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=5:00:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/postproc_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/postproc_%j.log

# ============================================================================
# UAAG Assay Post-Processing Script
# ============================================================================
# Runs post-processing for all iterations and splits of a single assay
# This should be run AFTER all 50 sampling tasks complete
# Usage: sbatch --dependency=afterok:<JOB_ID> run_assay_postprocess.sh <config_file>
# ============================================================================

echo "============================================================================"
echo "UAAG Post-Processing"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "============================================================================"

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_FILE=${1:-slurm_config/assays/ENVZ_ECOLI.txt}

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "Config file: ${CONFIG_FILE}"

# Read protein ID from the first data line (all rows have the same protein)
PROTEIN_ID=$(awk 'NR==2 {print $2}' ${CONFIG_FILE})
BASELINE=$(awk 'NR==2 {print $3}' ${CONFIG_FILE})

echo "Protein ID: ${PROTEIN_ID}"
echo "Baseline: ${BASELINE}"
echo ""

# Model and paths
MODEL=UAAG_model
NUM_SAMPLES=1000
WORK_DIR=/flash/project_465002574/UAAG2_main

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "→ Setting up environment..."

module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

cd ${WORK_DIR}

# ============================================================================
# POST-PROCESSING - Process each iteration
# ============================================================================
echo ""
echo "============================================================================"
echo "Running Post-Processing for All Iterations"
echo "============================================================================"

for ITERATION in {0..4}; do
    echo ""
    echo "[$(date)] Processing iteration ${ITERATION}..."
    
    # Process each split for this iteration
    for SPLIT in {0..9}; do
        RUN_ID="${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}"
        SAMPLES_DIR="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID}/Samples"
        
        if [ -d "${SAMPLES_DIR}" ]; then
            echo "  Processing split ${SPLIT}: ${SAMPLES_DIR}"
            python scripts/post_analysis.py --analysis_path ${SAMPLES_DIR}
            
            if [ $? -ne 0 ]; then
                echo "  ✗ Post-processing failed for iteration ${ITERATION}, split ${SPLIT}"
                # Continue processing other iterations even if one fails
            else
                echo "  ✓ Post-processing completed for iteration ${ITERATION}, split ${SPLIT}"
            fi
        else
            echo "  ⚠ Samples directory not found: ${SAMPLES_DIR}"
        fi
    done
    
    echo "[$(date)] ✓ Completed iteration ${ITERATION}"
done

# ============================================================================
# EVALUATION (Optional - combine results)
# ============================================================================
echo ""
echo "============================================================================"
echo "Evaluation and Results Aggregation"
echo "============================================================================"

# Find the first aa_distribution.csv to use for evaluation
FIRST_AA_DIST=$(find /scratch/project_465002574/ProteinGymSampling/run${MODEL}/${PROTEIN_ID}_* -name "aa_distribution.csv" | head -1)

if [ -n "${FIRST_AA_DIST}" ]; then
    echo "Found aa_distribution.csv: ${FIRST_AA_DIST}"
    
    OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}"
    mkdir -p ${OUTPUT_DIR}
    
    echo "[$(date)] Running evaluation..."
    python scripts/result_eval_uniform.py \
        --generated ${FIRST_AA_DIST} \
        --baselines /scratch/project_465002574/UNAAGI_benchmark_values/baselines/${BASELINE} \
        --total_num ${NUM_SAMPLES} \
        --output_dir ${OUTPUT_DIR}
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] ✓ Evaluation completed"
        echo "Results saved to: ${OUTPUT_DIR}"
    else
        echo "[$(date)] ⚠ Evaluation failed or baseline not available"
    fi
else
    echo "⚠ No aa_distribution.csv found - skipping evaluation"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "Post-Processing Completed!"
echo "============================================================================"
echo "End time: $(date)"
echo "Protein: ${PROTEIN_ID}"
echo "Iterations processed: 0-4 (5 total)"
echo "Splits per iteration: 0-9 (10 total)"
echo ""
echo "Results can be found in:"
echo "  /scratch/project_465002574/ProteinGymSampling/run${MODEL}/${PROTEIN_ID}_*/"
echo "============================================================================"
