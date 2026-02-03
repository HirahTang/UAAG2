#!/bin/bash
#SBATCH --job-name=PUMA_sampling
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9
#SBATCH -o logs/PUMA_job_%A_%a.log
#SBATCH -e logs/PUMA_job_%A_%a.log

echo "============================================================================"
echo "UAAG PUMA NCAA Benchmark Sampling"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "============================================================================"

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

rocm-smi || echo "Warning: rocm-smi not available"

git fetch origin
git checkout main
echo "Running on branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit hash:       $(git rev-parse HEAD)"

MODEL=UAAG_model
CKPT_PATH=/flash/project_465002574/${MODEL}/last.ckpt
BENCHMARK_PATH=/scratch/project_465002574/UNAAGI_benchmarks/2roc_puma.pt
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}
NUM_SAMPLES=1000
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15

# Construct unique run ID with split index
RUN_ID="${MODEL}/PUMA_${MODEL}_variational_sampling_${NUM_SAMPLES}_split${SPLIT_INDEX}"

echo "[$(date)] Starting PUMA sampling for ${RUN_ID}..."
echo "Split index: ${SPLIT_INDEX}"
echo "Benchmark: ${BENCHMARK_PATH}"

python scripts/generate_ligand.py \
    --load-ckpt ${CKPT_PATH} \
    --id ${RUN_ID} \
    --batch-size ${BATCH_SIZE} \
    --virtual_node_size ${VIRTUAL_NODE_SIZE} \
    --num-samples ${NUM_SAMPLES} \
    --benchmark-path ${BENCHMARK_PATH} \
    --split_index ${SPLIT_INDEX} \
    --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl

echo "[$(date)] PUMA sampling completed for ${RUN_ID}"

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ PUMA sampling completed successfully"
else
    echo "[$(date)] ✗ PUMA sampling failed with exit code $?"
    exit 1
fi

# Post-processing and analysis
SAMPLES_PATH="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID}/Samples"
OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/${MODEL}/PUMA_${MODEL}_variational_sampling_${NUM_SAMPLES}_split${SPLIT_INDEX}"

echo ""
echo "============================================================================"
echo "Starting PUMA Post-Processing and Analysis"
echo "============================================================================"
echo "[$(date)] Starting PUMA analysis for ${RUN_ID}..."

python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Post-processing completed successfully"
else
    echo "[$(date)] ✗ Post-processing failed with exit code $?"
    exit 1
fi

python scripts/result_eval_uniform_uaa.py \
    --benchmark /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/PUMA_reframe.csv \
    --aa_output ${SAMPLES_PATH}/aa_distribution.csv \
    --output_dir ${OUTPUT_DIR} \
    --total_num ${NUM_SAMPLES}

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ PUMA evaluation completed successfully"
else
    echo "[$(date)] ✗ PUMA evaluation failed with exit code $?"
    exit 1
fi

echo ""
echo "============================================================================"
echo "PUMA Analysis Completed Successfully!"
echo "============================================================================"
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================================"
