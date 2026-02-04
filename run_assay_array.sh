#!/bin/bash
#SBATCH --job-name=UAAG_array
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-49
#SBATCH -o /scratch/project_465002574/UAAG_logs/array_%A_%a.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/array_%A_%a.log

# ============================================================================
# UAAG Assay Array Job Script
# ============================================================================
# Runs all 50 tasks (5 iterations × 10 splits) for a single assay
# Usage: sbatch run_assay_array.sh <config_file>
# Example: sbatch run_assay_array.sh slurm_config/assays/ENVZ_ECOLI.txt
# ============================================================================

echo "============================================================================"
echo "UAAG Array Job - Task ${SLURM_ARRAY_TASK_ID}"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================================"

# ============================================================================
# CONFIGURATION
# ============================================================================
# Config file should be passed as first argument, or set default
CONFIG_FILE=${1:-slurm_config/assays/ENVZ_ECOLI.txt}

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "Config file: ${CONFIG_FILE}"

# Read configuration from the config file based on SLURM_ARRAY_TASK_ID
PROTEIN_ID=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} 'NR>1 && $1==ArrayID {print $2}' ${CONFIG_FILE})
BASELINE=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} 'NR>1 && $1==ArrayID {print $3}' ${CONFIG_FILE})
SPLIT_INDEX=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} 'NR>1 && $1==ArrayID {print $4}' ${CONFIG_FILE})
ITERATION=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} 'NR>1 && $1==ArrayID {print $5}' ${CONFIG_FILE})

# Verify we got the data
if [ -z "${PROTEIN_ID}" ]; then
    echo "Error: Could not read data for ArrayID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Protein ID: ${PROTEIN_ID}"
echo "Baseline: ${BASELINE}"
echo "Split: ${SPLIT_INDEX}"
echo "Iteration: ${ITERATION}"

# Model and paths
MODEL=UAAG_model
CKPT_PATH=/flash/project_465002574/UAAG2_main/${MODEL}/last.ckpt
BENCHMARK_PATH=/scratch/project_465002574/UNAAGI_benchmarks/${PROTEIN_ID}.pt
NUM_SAMPLES=1000
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15

# Output configuration
WORK_DIR=/flash/project_465002574/UAAG2_main
RUN_ID="${MODEL}/${PROTEIN_ID}_${MODEL}_${NUM_SAMPLES}_iter${ITERATION}"
SAVE_DIR=/scratch/project_465002574/ProteinGymSampling

echo "Benchmark: ${BENCHMARK_PATH}"
echo "Run ID: ${RUN_ID}"
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

cd ${WORK_DIR}

# Switch to main branch
git fetch origin
git checkout main

# Check GPU
echo ""
echo "→ Checking GPU availability..."
rocm-smi || echo "Warning: rocm-smi not available"

mkdir -p /scratch/project_465002574/UAAG_logs

# ============================================================================
# SAMPLING
# ============================================================================
echo ""
echo "============================================================================"
echo "Running Sampling - Iteration ${ITERATION}, Split ${SPLIT_INDEX}"
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
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "Task ${SLURM_ARRAY_TASK_ID} Completed Successfully!"
echo "============================================================================"
echo "End time: $(date)"
echo "Protein: ${PROTEIN_ID}"
echo "Iteration: ${ITERATION}"
echo "Split: ${SPLIT_INDEX}"
echo ""
echo "Note: Post-processing will run after all 50 tasks complete"
echo "============================================================================"
