#!/bin/bash

# ============================================================================
# UAAG Pipeline Test Script
# ============================================================================
# Quick validation test: 1 protein, 2 splits, 100 samples
# Tests the complete workflow without overwhelming the system
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "UAAG Pipeline Test Mode"
echo "============================================================================"
echo "Testing with: 1 protein (ENVZ_ECOLI), 2 splits, 100 samples"
echo "This will validate: sampling → analysis → compression"
echo "============================================================================"
echo ""

# Configuration
MODEL=UAAG_model
CKPT_PATH=/flash/project_465002574/UAAG2_main/${MODEL}/last.ckpt
CONFIG_FILE=/flash/project_465002574/UAAG2_main/slurm_config/slurm_config.txt
NUM_SAMPLES=100  # Reduced for testing
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15
TEST_PROTEIN_ID="ENVZ_ECOLI"
TEST_ITER="test"

# Verify checkpoint exists
if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

# Verify config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "✓ Checkpoint found: ${CKPT_PATH}"
echo "✓ Config file found: ${CONFIG_FILE}"
echo ""

# Create test script for 2 splits (array tasks 0-1)
TEST_SCRIPT="/tmp/test_sampling_${USER}.sh"
cat > ${TEST_SCRIPT} << 'EOF'
#!/bin/bash
#SBATCH --job-name=UAAG_test
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --array=40-41
#SBATCH -o /scratch/project_465002574/UAAG_logs/test_pipeline_%A_%a.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/test_pipeline_%A_%a.log

echo "============================================================================"
echo "UAAG Pipeline Test - Sampling"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================================================"

module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

cd /flash/project_465002574/UAAG2_main

MODEL=MODEL_PLACEHOLDER
CKPT_PATH=CKPT_PLACEHOLDER
CONFIG_FILE=CONFIG_FILE_PLACEHOLDER
NUM_SAMPLES=NUM_SAMPLES_PLACEHOLDER
BATCH_SIZE=BATCH_SIZE_PLACEHOLDER
VIRTUAL_NODE_SIZE=VIRTUAL_NODE_SIZE_PLACEHOLDER

ID=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG_FILE})
SPLIT_INDEX=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG_FILE})
BENCHMARK_PATH=/scratch/project_465002574/UNAAGI_benchmarks/${ID}.pt

RUN_ID="${MODEL}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_test_split${SPLIT_INDEX}"

echo "Protein ID: ${ID}"
echo "Split index: ${SPLIT_INDEX}"
echo "Benchmark: ${BENCHMARK_PATH}"
echo "Samples: ${NUM_SAMPLES}"
echo ""

# Check if benchmark exists
if [ ! -f "${BENCHMARK_PATH}" ]; then
    echo "ERROR: Benchmark file not found: ${BENCHMARK_PATH}"
    exit 1
fi

echo "[$(date)] Starting sampling..."
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
    echo "[$(date)] ✗ Sampling failed"
    exit 1
fi

echo "============================================================================"
EOF

# Replace placeholders
sed -i "s|MODEL_PLACEHOLDER|${MODEL}|g" ${TEST_SCRIPT}
sed -i "s|CKPT_PLACEHOLDER|${CKPT_PATH}|g" ${TEST_SCRIPT}
sed -i "s|CONFIG_FILE_PLACEHOLDER|${CONFIG_FILE}|g" ${TEST_SCRIPT}
sed -i "s|NUM_SAMPLES_PLACEHOLDER|${NUM_SAMPLES}|g" ${TEST_SCRIPT}
sed -i "s|BATCH_SIZE_PLACEHOLDER|${BATCH_SIZE}|g" ${TEST_SCRIPT}
sed -i "s|VIRTUAL_NODE_SIZE_PLACEHOLDER|${VIRTUAL_NODE_SIZE}|g" ${TEST_SCRIPT}

echo "Submitting test job..."
echo ""
JOB_ID=$(sbatch --parsable ${TEST_SCRIPT} 2>&1)

if [[ $JOB_ID =~ ^[0-9]+$ ]]; then
    echo "✓ Test sampling job submitted: ${JOB_ID}"
    echo ""
    echo "Monitor with: squeue -j ${JOB_ID}"
    echo ""
    
    # Create analysis script that runs after sampling completes
    ANALYSIS_SCRIPT="/tmp/test_analysis_${USER}.sh"
    cat > ${ANALYSIS_SCRIPT} << 'ANALYSIS_EOF'
#!/bin/bash
#SBATCH --job-name=UAAG_test_analysis
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=30:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/test_analysis_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/test_analysis_%j.log

echo "============================================================================"
echo "UAAG Pipeline Test - Analysis & Compression"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================================"

module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

cd /flash/project_465002574/UAAG2_main

MODEL=MODEL_PLACEHOLDER
CONFIG_FILE=CONFIG_FILE_PLACEHOLDER
NUM_SAMPLES=NUM_SAMPLES_PLACEHOLDER

# Get protein ID from the first array task (40) used in sampling
PROTEIN_ID=$(awk -v ArrayID=40 '$1==ArrayID {print $2; exit}' ${CONFIG_FILE})
BASELINE=$(awk -v ID="${PROTEIN_ID}" '$2==ID {print $3; exit}' ${CONFIG_FILE})

echo "Protein: ${PROTEIN_ID} (from config array task 40)"
echo "Baseline: ${BASELINE}"
echo ""

RUN_ID_BASE="${MODEL}/${PROTEIN_ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_test"
SAMPLES_DIR="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID_BASE}_split0/Samples"
OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/TEST/${PROTEIN_ID}_test"

# Step 1: Post-processing
echo "[$(date)] Step 1: Running post-processing..."
SAMPLES_PATH="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID_BASE}_split*/Samples"
python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Post-processing completed"
else
    echo "[$(date)] ✗ Post-processing failed"
    exit 1
fi

# Step 2: Evaluation
echo ""
echo "[$(date)] Step 2: Running evaluation..."
python scripts/result_eval_uniform.py \
    --generated ${SAMPLES_DIR}/aa_distribution.csv \
    --baselines /scratch/project_465002574/UNAAGI_benchmark_values/baselines/${BASELINE} \
    --total_num ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR}

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Evaluation completed"
    echo "Results saved to: ${OUTPUT_DIR}"
else
    echo "[$(date)] ✗ Evaluation failed"
    exit 1
fi

# Step 3: Compression
echo ""
echo "[$(date)] Step 3: Compressing samples..."
ARCHIVE_DIR="/scratch/project_465002574/UNAAGI_archives/TEST"
mkdir -p ${ARCHIVE_DIR}
ARCHIVE_NAME="${ARCHIVE_DIR}/${PROTEIN_ID}_test.tar.gz"

cd /scratch/project_465002574/ProteinGymSampling/
tar -czf ${ARCHIVE_NAME} run${RUN_ID_BASE}_split*/

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Archive created: ${ARCHIVE_NAME}"
    echo "Archive size: $(du -h ${ARCHIVE_NAME} | cut -f1)"
    
    # Remove original directories
    echo "[$(date)] Removing original directories..."
    rm -rf run${RUN_ID_BASE}_split*/
    echo "[$(date)] ✓ Cleanup completed"
else
    echo "[$(date)] ✗ Archive creation failed"
    exit 1
fi

echo ""
echo "============================================================================"
echo "TEST PIPELINE COMPLETED SUCCESSFULLY!"
echo "============================================================================"
echo "Results:"
echo "  - Evaluation: ${OUTPUT_DIR}"
echo "  - Archive: ${ARCHIVE_NAME}"
echo ""
echo "All systems validated! Ready for full pipeline:"
echo "  bash run_pipeline_monitored.sh"
echo "============================================================================"
ANALYSIS_EOF

    # Replace placeholders
    sed -i "s|MODEL_PLACEHOLDER|${MODEL}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|CONFIG_FILE_PLACEHOLDER|${CONFIG_FILE}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|NUM_SAMPLES_PLACEHOLDER|${NUM_SAMPLES}|g" ${ANALYSIS_SCRIPT}
    
    # Submit analysis job with dependency on sampling
    echo "Submitting test analysis job (runs after sampling completes)..."
    ANALYSIS_JOB=$(sbatch --parsable --dependency=afterok:${JOB_ID} ${ANALYSIS_SCRIPT} 2>&1)
    
    if [[ $ANALYSIS_JOB =~ ^[0-9]+$ ]]; then
        echo "✓ Test analysis job submitted: ${ANALYSIS_JOB}"
        echo ""
        echo "============================================================================"
        echo "Test Pipeline Submitted"
        echo "============================================================================"
        echo "Sampling job: ${JOB_ID} (running)"
        echo "Analysis job: ${ANALYSIS_JOB} (pending, waits for sampling)"
        echo ""
        echo "Monitor progress:"
        echo "  squeue -j ${JOB_ID},${ANALYSIS_JOB}"
        echo ""
        echo "View logs:"
        echo "  tail -f /scratch/project_465002574/UAAG_logs/test_pipeline_${JOB_ID}_*.log"
        echo "  tail -f /scratch/project_465002574/UAAG_logs/test_analysis_${ANALYSIS_JOB}.log"
        echo ""
        echo "When complete, if successful, run full pipeline with:"
        echo "  bash run_pipeline_monitored.sh"
        echo "============================================================================"
    else
        echo "✗ Analysis job submission failed: ${ANALYSIS_JOB}"
        exit 1
    fi
else
    echo "✗ Job submission failed: ${JOB_ID}"
    exit 1
fi
