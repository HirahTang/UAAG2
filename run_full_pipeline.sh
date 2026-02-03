#!/bin/bash

# ============================================================================
# UAAG Full Pipeline Submission Script
# ============================================================================
# This script submits separate SLURM jobs for sampling (iterations 1-5) 
# and analysis. Each iteration runs as an independent job with dependencies.
# Only requires MODEL path to be specified.
# ============================================================================

# ============================================================================
# CONFIGURATION - ONLY EDIT THIS SECTION
# ============================================================================
MODEL=Full_mask_5_virtual_node_mask_token_atomic_only_mask_diffusion_0917
CKPT_PATH=/home/qcx679/hantang/UAAG2/3DcoordsAtomsBonds_0/run${MODEL}/last.ckpt
CONFIG_FILE=/home/qcx679/hantang/UAAG2/slurm_config/slurm_config.txt
NUM_SAMPLES=1000
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15
TOTAL_NUM=1000  # For analysis script

# SLURM settings for sampling jobs
SAMPLING_TIME="2-00:00:00"
SAMPLING_PARTITION="gpu,boomsma"
SAMPLING_EXCLUDE="hendrixgpu01fl,hendrixgpu16fl,hendrixgpu19fl,hendrixgpu04fl,hendrixgpu26fl,hendrixgpu24fl,hendrixgpu25fl,hendrixgpu06fl"
SAMPLING_ARRAY="0-249%9"  # 250 array jobs (25 proteins × 10 splits)

# SLURM settings for analysis jobs
ANALYSIS_TIME="5:00:00"
ANALYSIS_PARTITION="gpu,boomsma"
ANALYSIS_ARRAY="0-24"  # 25 analysis jobs (1 per protein, combining all 10 splits)
# ============================================================================

echo "============================================================================"
echo "UAAG Full Pipeline Job Submission"
echo "============================================================================"
echo "Model: ${MODEL}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Config file: ${CONFIG_FILE}"
echo "Number of samples per iteration: ${NUM_SAMPLES}"
echo "Array jobs per iteration: 250 sampling (25 proteins × 10 splits), 25 analysis (1 per protein)"
echo "Iterations: 1-5"
echo "Total jobs: 1,395 (1,250 sampling + 125 analysis + 20 UAA benchmarks)"
echo "============================================================================"

# Create temporary script directory
SCRIPT_DIR="/home/qcx679/hantang/UAAG2/tmp_scripts"
mkdir -p ${SCRIPT_DIR}

# Loop through 5 iterations and submit jobs
for i in {1..5}; do
    echo ""
    echo "Submitting jobs for iteration ${i}..."
    
    # Create sampling script
    SAMPLING_SCRIPT="${SCRIPT_DIR}/sampling_iter${i}.sh"
    cat > ${SAMPLING_SCRIPT} << 'SAMPLING_EOF'
#!/bin/bash
#SBATCH --job-name=UAAG_samp_ITER
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --partition=SAMPLING_PARTITION_PLACEHOLDER
#SBATCH --time=SAMPLING_TIME_PLACEHOLDER
#SBATCH --array=SAMPLING_ARRAY_PLACEHOLDER
#SBATCH --exclude=SAMPLING_EXCLUDE_PLACEHOLDER
#SBATCH -o logs/sampling_iter_ITER_%A_%a.log
#SBATCH -e logs/sampling_iter_ITER_%A_%a.log

nvidia-smi
echo "Job $SLURM_JOB_ID is running on node: $SLURMD_NODENAME"
source ~/.bashrc
conda activate targetdiff
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qcx679/.conda/envs/targetdiff/lib

git fetch origin
git checkout main

MODEL=MODEL_PLACEHOLDER
CKPT_PATH=CKPT_PLACEHOLDER
CONFIG_FILE=CONFIG_FILE_PLACEHOLDER
NUM_SAMPLES=NUM_SAMPLES_PLACEHOLDER
BATCH_SIZE=BATCH_SIZE_PLACEHOLDER
VIRTUAL_NODE_SIZE=VIRTUAL_NODE_SIZE_PLACEHOLDER

# Read from config file using SLURM_ARRAY_TASK_ID
ID=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG_FILE})
SPLIT_INDEX=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG_FILE})
BENCHMARK_PATH=/home/qcx679/hantang/UAAG2/data/full_graph/benchmarks/${ID}.pt

# Construct unique run ID with protein ID and split index
RUN_ID="${MODEL}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER_split${SPLIT_INDEX}"

echo "[$(date)] Starting sampling for ${RUN_ID}..."
echo "Protein ID: ${ID}"
echo "Split index: ${SPLIT_INDEX}"
echo "Benchmark: ${BENCHMARK_PATH}"

python scripts/generate_ligand.py \
    --load-ckpt ${CKPT_PATH} \
    --id ${RUN_ID} \
    --batch-size ${BATCH_SIZE} \
    --virtual_node_size ${VIRTUAL_NODE_SIZE} \
    --num-samples ${NUM_SAMPLES} \
    --benchmark-path ${BENCHMARK_PATH} \
    --split_index ${SPLIT_INDEX}

echo "[$(date)] Sampling completed for ${RUN_ID}"
SAMPLING_EOF

    # Replace placeholders in sampling script
    sed -i "s|ITER|${i}|g" ${SAMPLING_SCRIPT}
    sed -i "s|MODEL_PLACEHOLDER|${MODEL}|g" ${SAMPLING_SCRIPT}
    sed -i "s|CKPT_PLACEHOLDER|${CKPT_PATH}|g" ${SAMPLING_SCRIPT}
    sed -i "s|CONFIG_FILE_PLACEHOLDER|${CONFIG_FILE}|g" ${SAMPLING_SCRIPT}
    sed -i "s|NUM_SAMPLES_PLACEHOLDER|${NUM_SAMPLES}|g" ${SAMPLING_SCRIPT}
    sed -i "s|BATCH_SIZE_PLACEHOLDER|${BATCH_SIZE}|g" ${SAMPLING_SCRIPT}
    sed -i "s|VIRTUAL_NODE_SIZE_PLACEHOLDER|${VIRTUAL_NODE_SIZE}|g" ${SAMPLING_SCRIPT}
    sed -i "s|SAMPLING_TIME_PLACEHOLDER|${SAMPLING_TIME}|g" ${SAMPLING_SCRIPT}
    sed -i "s|SAMPLING_PARTITION_PLACEHOLDER|${SAMPLING_PARTITION}|g" ${SAMPLING_SCRIPT}
    sed -i "s|SAMPLING_EXCLUDE_PLACEHOLDER|${SAMPLING_EXCLUDE}|g" ${SAMPLING_SCRIPT}
    sed -i "s|SAMPLING_ARRAY_PLACEHOLDER|${SAMPLING_ARRAY}|g" ${SAMPLING_SCRIPT}
    
    # Submit sampling job
    SAMPLING_JOB=$(sbatch --parsable ${SAMPLING_SCRIPT})
    echo "  Sampling job submitted: ${SAMPLING_JOB}"
    
    # Create analysis script
    ANALYSIS_SCRIPT="${SCRIPT_DIR}/analysis_iter${i}.sh"
    cat > ${ANALYSIS_SCRIPT} << 'ANALYSIS_EOF'
#!/bin/bash
#SBATCH --job-name=UAAG_anal_ITER
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=ANALYSIS_PARTITION_PLACEHOLDER
#SBATCH --time=ANALYSIS_TIME_PLACEHOLDER
#SBATCH --array=ANALYSIS_ARRAY_PLACEHOLDER
#SBATCH --exclude=SAMPLING_EXCLUDE_PLACEHOLDER
#SBATCH -o logs/analysis_iter_ITER_%A_%a.log
#SBATCH -e logs/analysis_iter_ITER_%A_%a.log

source ~/.bashrc
conda activate targetdiff
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qcx679/.conda/envs/targetdiff/lib

git fetch origin
git checkout main
echo "Running on branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit hash:       $(git rev-parse HEAD)"

MODEL=MODEL_PLACEHOLDER
CONFIG_FILE=CONFIG_FILE_PLACEHOLDER
NUM_SAMPLES=NUM_SAMPLES_PLACEHOLDER
TOTAL_NUM=TOTAL_NUM_PLACEHOLDER

# Read protein ID from config file - each analysis task handles one protein (array ID 0-24)
# We need to find the first occurrence of each protein in the config
ID=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} 'BEGIN{count=0} {if(count==ArrayID){print $2; exit} if(NR>1 && prev!=$2){count++} prev=$2}' ${CONFIG_FILE})
BASELINE=$(awk -v ID="${ID}" '$2==ID {print $3; exit}' ${CONFIG_FILE})

# Construct run ID pattern (analysis combines all splits 0-9)
RUN_ID_BASE="${MODEL}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER"
SAMPLES_PATH="/datasets/biochem/unaagi/ProteinGymSampling/run${RUN_ID_BASE}_split*/Samples"
OUTPUT_DIR="/home/qcx679/hantang/UAAG2/results/${MODEL}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER"

echo "[$(date)] Starting analysis for protein ${ID}..."
echo "Protein ID: ${ID}"
echo "Baseline: ${BASELINE}"
echo "Combining splits: 0-9"

# Step 1: Post-processing to generate aa_distribution.csv
echo "[$(date)] Running post-processing..."
python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}

# Step 2: Evaluation against baselines
echo "[$(date)] Running evaluation..."
python scripts/result_eval_uniform.py \
    --generated ${SAMPLES_PATH}/aa_distribution.csv \
    --baselines /home/qcx679/hantang/UAAG2/data/baselines/${BASELINE} \
    --total_num ${TOTAL_NUM} \
    --output_dir ${OUTPUT_DIR}

echo "[$(date)] Analysis completed for ${RUN_ID_BASE}"
echo "Results saved to: ${OUTPUT_DIR}"
ANALYSIS_EOF

    # Replace placeholders in analysis script
    sed -i "s|ITER|${i}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|MODEL_PLACEHOLDER|${MODEL}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|CONFIG_FILE_PLACEHOLDER|${CONFIG_FILE}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|NUM_SAMPLES_PLACEHOLDER|${NUM_SAMPLES}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|TOTAL_NUM_PLACEHOLDER|${TOTAL_NUM}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|ANALYSIS_TIME_PLACEHOLDER|${ANALYSIS_TIME}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|ANALYSIS_PARTITION_PLACEHOLDER|${ANALYSIS_PARTITION}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|SAMPLING_EXCLUDE_PLACEHOLDER|${SAMPLING_EXCLUDE}|g" ${ANALYSIS_SCRIPT}
    sed -i "s|ANALYSIS_ARRAY_PLACEHOLDER|${ANALYSIS_ARRAY}|g" ${ANALYSIS_SCRIPT}
    
    # Submit analysis job with dependency on all 10 sampling splits for each protein
    # Analysis array task N waits for sampling array tasks N*10 through N*10+9
    ANALYSIS_JOB=$(sbatch --parsable --dependency=afterok:${SAMPLING_JOB} ${ANALYSIS_SCRIPT})
    echo "  Analysis job submitted: ${ANALYSIS_JOB} (waits for all 250 sampling tasks in ${SAMPLING_JOB})"
done

echo ""
echo "============================================================================"
echo "Submitting CP2 and PUMA UAA benchmark jobs..."
echo "============================================================================"

# Submit CP2 jobs (10 array jobs, no iterations)
echo "Submitting CP2 jobs..."
CP2_JOB=$(sbatch --parsable run_sampling_CP2_sbatch.sh)
echo "  CP2 job submitted: ${CP2_JOB}"

# Submit PUMA jobs (10 array jobs, no iterations)
echo "Submitting PUMA jobs..."
PUMA_JOB=$(sbatch --parsable run_sampling_PUMA_sbatch.sh)
echo "  PUMA job submitted: ${PUMA_JOB}"

echo ""
echo "============================================================================"
echo "All jobs submitted successfully!"
echo "============================================================================"
echo "ProteinGym iterations: 1-5 (1,375 jobs)"
echo "  - Sampling jobs: 1,250 (250 per iteration)"
echo "  - Analysis jobs: 125 (25 per iteration, 1 per protein combining all 10 splits)"
echo "UAA benchmarks:"
echo "  - CP2: ${CP2_JOB} (10 array jobs)"
echo "  - PUMA: ${PUMA_JOB} (10 array jobs)"
echo "Total jobs: 1,395"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: logs/"
echo "Temporary scripts in: ${SCRIPT_DIR}"
echo "============================================================================"

python scripts/result_eval_uniform.py \
    --generated /scratch/project_465002574/ProteinGymSampling/runUAAG_model/ENVZ_ECOLI_test_100_samples/Samples/aa_distribution.csv \
    --baselines /scratch/project_465002574/UNAAGI_benchmark_values/baselines/ENVZ_ECOLI_Ghose_2023.csv \
    --total_num 100 \
    --output_dir /scratch/project_465002574/UNAAGI_result/test/