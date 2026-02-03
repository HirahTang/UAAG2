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
MODEL=UAAG_model
CKPT_PATH=/flash/project_465002574/${MODEL}/last.ckpt
CONFIG_FILE=/flash/project_465002574/UAAG2_main/slurm_config/slurm_config.txt
NUM_SAMPLES=1000
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15
TOTAL_NUM=1000  # For analysis script

# SLURM settings for sampling jobs
SAMPLING_TIME="2-00:00:00"
SAMPLING_PARTITION="standard-g"
SAMPLING_ARRAY="0-249%9"  # 250 array jobs (25 proteins × 10 splits)

# SLURM settings for analysis jobs
ANALYSIS_TIME="5:00:00"
ANALYSIS_PARTITION="standard-g"
# ============================================================================

echo "============================================================================"
echo "UAAG Full Pipeline Job Submission"
echo "============================================================================"
echo "Model: ${MODEL}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Config file: ${CONFIG_FILE}"
echo "Number of samples per iteration: ${NUM_SAMPLES}"
echo "Jobs per iteration: 250 sampling + 25 analysis (per-protein)"
echo "Iterations: 1-5"
echo "Total jobs: 1,395 (1,250 sampling + 125 analysis + 20 UAA benchmarks)"
echo ""
echo "Workflow per protein:"
echo "  1. Sample (10 splits in parallel)"
echo "  2. Post-process & evaluate (as soon as sampling completes)"
echo "  3. Compress samples into archive"
echo "============================================================================"

# Create temporary script directory
SCRIPT_DIR="/flash/project_465002574/UAAG2_main/tmp_scripts"
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
#SBATCH --account=project_465002574
#SBATCH --partition=SAMPLING_PARTITION_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=SAMPLING_TIME_PLACEHOLDER
#SBATCH --array=SAMPLING_ARRAY_PLACEHOLDER
#SBATCH -o logs/sampling_iter_ITER_%A_%a.log
#SBATCH -e logs/sampling_iter_ITER_%A_%a.log

rocm-smi || echo "Warning: rocm-smi not available"
echo "Job $SLURM_JOB_ID is running on node: $SLURMD_NODENAME"
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

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
BENCHMARK_PATH=/scratch/project_465002574/UNAAGI_benchmarks/${ID}.pt

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
    --split_index ${SPLIT_INDEX} \
    --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl

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
    sed -i "s|SAMPLING_ARRAY_PLACEHOLDER|${SAMPLING_ARRAY}|g" ${SAMPLING_SCRIPT}
    
    # Submit sampling job
    SAMPLING_JOB=$(sbatch --parsable ${SAMPLING_SCRIPT})
    echo "  Sampling job submitted: ${SAMPLING_JOB}"
    
    # Submit individual analysis jobs per protein (25 proteins)
    # Each analysis job depends only on its 10 sampling splits
    echo "  Submitting 25 per-protein analysis jobs..."
    
    for protein_idx in {0..24}; do
        # Create per-protein analysis script
        ANALYSIS_SCRIPT="${SCRIPT_DIR}/analysis_iter${i}_protein${protein_idx}.sh"
        cat > ${ANALYSIS_SCRIPT} << 'ANALYSIS_EOF'
#!/bin/bash
#SBATCH --job-name=UAAG_anal_ITER_PROT
#SBATCH --account=project_465002574
#SBATCH --partition=ANALYSIS_PARTITION_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=ANALYSIS_TIME_PLACEHOLDER
#SBATCH -o logs/analysis_iter_ITER_PROT_%j.log
#SBATCH -e logs/analysis_iter_ITER_PROT_%j.log

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

git fetch origin
git checkout main
echo "Running on branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit hash:       $(git rev-parse HEAD)"

MODEL=MODEL_PLACEHOLDER
CONFIG_FILE=CONFIG_FILE_PLACEHOLDER
NUM_SAMPLES=NUM_SAMPLES_PLACEHOLDER
TOTAL_NUM=TOTAL_NUM_PLACEHOLDER
PROTEIN_IDX=PROTEIN_IDX_PLACEHOLDER

# Read protein ID from config file
ID=$(awk -v ArrayID=${PROTEIN_IDX} 'BEGIN{count=0} {if(count==ArrayID){print $2; exit} if(NR>1 && prev!=$2){count++} prev=$2}' ${CONFIG_FILE})
BASELINE=$(awk -v ID="${ID}" '$2==ID {print $3; exit}' ${CONFIG_FILE})

# Construct run ID pattern (analysis combines all splits 0-9)
RUN_ID_BASE="${MODEL}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER"
SAMPLES_DIR="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID_BASE}_split0/Samples"
OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/${MODEL}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER"

echo "============================================================================"
echo "UAAG Analysis - Protein ${ID} (Iteration ITER)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Protein ID: ${ID}"
echo "Protein index: ${PROTEIN_IDX}"
echo "Baseline: ${BASELINE}"
echo "Combining splits: 0-9"
echo "Start time: $(date)"
echo "============================================================================"

# Step 1: Post-processing to generate aa_distribution.csv
echo ""
echo "[$(date)] Step 1: Running post-processing..."
SAMPLES_PATH="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID_BASE}_split*/Samples"
python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Post-processing completed successfully"
else
    echo "[$(date)] ✗ Post-processing failed with exit code $?"
    exit 1
fi

# Step 2: Evaluation against baselines
echo ""
echo "[$(date)] Step 2: Running evaluation..."
python scripts/result_eval_uniform.py \
    --generated ${SAMPLES_DIR}/aa_distribution.csv \
    --baselines /scratch/project_465002574/UNAAGI_benchmark_values/baselines/${BASELINE} \
    --total_num ${TOTAL_NUM} \
    --output_dir ${OUTPUT_DIR}

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Evaluation completed successfully"
else
    echo "[$(date)] ✗ Evaluation failed with exit code $?"
    exit 1
fi

# Step 3: Compress sampled structures
echo ""
echo "[$(date)] Step 3: Compressing sampled structures..."
ARCHIVE_DIR="/scratch/project_465002574/UNAAGI_archives/${MODEL}"
mkdir -p ${ARCHIVE_DIR}

# Compress all 10 splits for this protein
ARCHIVE_NAME="${ARCHIVE_DIR}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER.tar.gz"
echo "Creating archive: ${ARCHIVE_NAME}"

cd /scratch/project_465002574/ProteinGymSampling/
tar -czf ${ARCHIVE_NAME} run${RUN_ID_BASE}_split*/

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Archive created successfully"
    echo "Archive size: $(du -h ${ARCHIVE_NAME} | cut -f1)"
    
    # Optional: Remove original directories to save space
    # Uncomment the following lines if you want to delete after archiving
    # echo "[$(date)] Removing original sample directories..."
    # rm -rf run${RUN_ID_BASE}_split*/
    # echo "[$(date)] ✓ Original directories removed"
else
    echo "[$(date)] ✗ Archive creation failed with exit code $?"
    exit 1
fi

echo ""
echo "============================================================================"
echo "Analysis Completed Successfully!"
echo "============================================================================"
echo "End time: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "Archive: ${ARCHIVE_NAME}"
echo "============================================================================"
ANALYSIS_EOF

        # Replace placeholders in analysis script
        sed -i "s|ITER|${i}|g" ${ANALYSIS_SCRIPT}
        sed -i "s|PROT|${protein_idx}|g" ${ANALYSIS_SCRIPT}
        sed -i "s|MODEL_PLACEHOLDER|${MODEL}|g" ${ANALYSIS_SCRIPT}
        sed -i "s|CONFIG_FILE_PLACEHOLDER|${CONFIG_FILE}|g" ${ANALYSIS_SCRIPT}
        sed -i "s|NUM_SAMPLES_PLACEHOLDER|${NUM_SAMPLES}|g" ${ANALYSIS_SCRIPT}
        sed -i "s|TOTAL_NUM_PLACEHOLDER|${TOTAL_NUM}|g" ${ANALYSIS_SCRIPT}
        sed -i "s|ANALYSIS_TIME_PLACEHOLDER|${ANALYSIS_TIME}|g" ${ANALYSIS_SCRIPT}
        sed -i "s|ANALYSIS_PARTITION_PLACEHOLDER|${ANALYSIS_PARTITION}|g" ${ANALYSIS_SCRIPT}
        sed -i "s|PROTEIN_IDX_PLACEHOLDER|${protein_idx}|g" ${ANALYSIS_SCRIPT}
        
        # Calculate sampling array task IDs for this protein (10 splits)
        start_task=$((protein_idx * 10))
        end_task=$((start_task + 9))
        
        # Build dependency string for only this protein's 10 sampling tasks
        dep_string="afterok"
        for task_id in $(seq ${start_task} ${end_task}); do
            dep_string="${dep_string}:${SAMPLING_JOB}_${task_id}"
        done
        
        # Submit analysis job with dependency on only this protein's 10 sampling splits
        ANALYSIS_JOB=$(sbatch --parsable --dependency=${dep_string} ${ANALYSIS_SCRIPT})
        
        if [ $((protein_idx % 5)) -eq 0 ]; then
            echo "    Protein ${protein_idx}: ${ANALYSIS_JOB} (depends on sampling tasks ${start_task}-${end_task})"
        fi
    done
    echo "  ✓ All 25 analysis jobs submitted for iteration ${i}"
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