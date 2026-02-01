#!/bin/bash
#SBATCH --job-name=UAAG_test_ENVZ
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH -o logs/test_ENVZ_%j.log
#SBATCH -e logs/test_ENVZ_%j.log

# ============================================================================
# UAAG Single Assay Test Script for LUMI
# ============================================================================
# Tests a single protein assay (ENVZ_ECOLI) with the UAAG pipeline
# ============================================================================

echo "============================================================================"
echo "UAAG Single Assay Test - ENVZ_ECOLI"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================================"

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK_PATH=/flash/project_465002574/UAAG2_main/ENVZ_ECOLI.pt
MODEL=UAAG_model
CKPT_PATH=/flash/project_465002574/UAAG2_main/${MODEL}/last.ckpt
NUM_SAMPLES=100  # Reduced for testing
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15
SPLIT_INDEX=0
PROTEIN_ID=ENVZ_ECOLI

# Output paths
WORK_DIR=/flash/project_465002574/UAAG2_main
RUN_ID="${MODEL}/${PROTEIN_ID}_test_${NUM_SAMPLES}_samples"
OUTPUT_DIR="${WORK_DIR}/results/${RUN_ID}"
SAMPLES_DIR="${WORK_DIR}/outputs/${RUN_ID}/Samples"

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

echo "Benchmark: ${BENCHMARK_PATH}"
echo "Model: ${MODEL}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Samples: ${NUM_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "→ Setting up environment..."

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

# Activate your environment (adjust path as needed)
# If using conda-containerize or singularity:
# singularity exec --bind /flash:/flash container.sif bash -c "..."
# For now, assuming uv-based environment from hantang_env
cd ${WORK_DIR}

# Check GPU
echo ""
echo "→ Checking GPU availability..."
rocm-smi || echo "Warning: rocm-smi not available"

# ============================================================================
# STEP 1: SAMPLING
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 1: Running Sampling"
echo "============================================================================"
echo "[$(date)] Starting sampling for ${PROTEIN_ID}..."

python scripts/generate_ligand.py \
    --load-ckpt ${CKPT_PATH} \
    --id ${RUN_ID} \
    --batch-size ${BATCH_SIZE} \
    --virtual_node_size ${VIRTUAL_NODE_SIZE} \
    --num-samples ${NUM_SAMPLES} \
    --benchmark-path ${BENCHMARK_PATH} \
    --split_index ${SPLIT_INDEX} \
    --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Sampling completed successfully"
else
    echo "[$(date)] ✗ Sampling failed with exit code $?"
    exit 1
fi

# ============================================================================
# STEP 2: POST-PROCESSING
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 2: Running Post-Processing"
echo "============================================================================"
echo "[$(date)] Generating aa_distribution.csv..."

python scripts/post_analysis.py --analysis_path ${SAMPLES_DIR}

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Post-processing completed successfully"
else
    echo "[$(date)] ✗ Post-processing failed with exit code $?"
    exit 1
fi

# ============================================================================
# STEP 3: EVALUATION (Optional - if baseline available)
# ============================================================================
# Uncomment if you have baseline data for ENVZ_ECOLI
# echo ""
# echo "============================================================================"
# echo "STEP 3: Running Evaluation"
# echo "============================================================================"
# BASELINE_PATH=/flash/project_465002574/UAAG2_main/data/baselines/ENVZ_ECOLI_baseline.csv
# 
# if [ -f "${BASELINE_PATH}" ]; then
#     echo "[$(date)] Running evaluation against baseline..."
#     python scripts/result_eval_uniform.py \
#         --generated ${SAMPLES_DIR}/aa_distribution.csv \
#         --baselines ${BASELINE_PATH} \
#         --total_num ${NUM_SAMPLES} \
#         --output_dir ${OUTPUT_DIR}
#     
#     if [ $? -eq 0 ]; then
#         echo "[$(date)] ✓ Evaluation completed successfully"
#     else
#         echo "[$(date)] ✗ Evaluation failed with exit code $?"
#     fi
# else
#     echo "[$(date)] ⚠ Baseline not found, skipping evaluation"
# fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "Test Completed Successfully!"
echo "============================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Samples: ${SAMPLES_DIR}"
echo "  - AA Distribution: ${SAMPLES_DIR}/aa_distribution.csv"
echo "  - Output: ${OUTPUT_DIR}"
echo ""
echo "To check results:"
echo "  head ${SAMPLES_DIR}/aa_distribution.csv"
echo "============================================================================"
