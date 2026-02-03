#!/bin/bash

# ============================================================================
# UAAG Monitored Pipeline Submission Script
# ============================================================================
# This script monitors your job queue and submits new jobs when capacity 
# is available. Designed to run in tmux to work around job submission limits.
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL=UAAG_model
CKPT_PATH=/flash/project_465002574/${MODEL}/last.ckpt
CONFIG_FILE=/flash/project_465002574/UAAG2_main/slurm_config/slurm_config.txt
NUM_SAMPLES=1000
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15
TOTAL_NUM=1000

# Job limits
MAX_JOBS_ALLOWED=50  # Adjust based on your limit (check with: sacctmgr show qos)
CHECK_INTERVAL=300   # Check every 5 minutes (300 seconds)

# SLURM settings
SAMPLING_TIME="2-00:00:00"
SAMPLING_PARTITION="standard-g"
SAMPLING_ARRAY="0-249%9"
ANALYSIS_TIME="5:00:00"
ANALYSIS_PARTITION="standard-g"

# Tracking files
SCRIPT_DIR="/flash/project_465002574/UAAG2_main/tmp_scripts"
STATE_FILE="${SCRIPT_DIR}/pipeline_state.txt"
LOG_FILE="${SCRIPT_DIR}/pipeline_monitor.log"

mkdir -p ${SCRIPT_DIR}

# ============================================================================
# FUNCTIONS
# ============================================================================

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${LOG_FILE}
}

get_current_job_count() {
    squeue -u $USER -h -t pending,running | wc -l
}

wait_for_capacity() {
    while true; do
        CURRENT_JOBS=$(get_current_job_count)
        if [ $CURRENT_JOBS -lt $MAX_JOBS_ALLOWED ]; then
            log_message "Current jobs: ${CURRENT_JOBS}/${MAX_JOBS_ALLOWED} - capacity available"
            return 0
        else
            log_message "Current jobs: ${CURRENT_JOBS}/${MAX_JOBS_ALLOWED} - waiting ${CHECK_INTERVAL}s..."
            sleep ${CHECK_INTERVAL}
        fi
    done
}

# ============================================================================
# INITIALIZE STATE
# ============================================================================

if [ ! -f ${STATE_FILE} ]; then
    cat > ${STATE_FILE} << EOF
# Pipeline state tracking
# Format: iteration,stage,job_id,status
# Stages: sampling_pending, sampling_running, analysis_pending, analysis_running, completed
# Status: pending, submitted, running, completed, failed
EOF
    
    # Initialize all iterations as pending
    for i in {1..5}; do
        echo "${i},sampling_pending,," >> ${STATE_FILE}
    done
    echo "cp2,sampling_pending,," >> ${STATE_FILE}
    echo "puma,sampling_pending,," >> ${STATE_FILE}
fi

log_message "============================================================================"
log_message "UAAG Monitored Pipeline Starting"
log_message "============================================================================"
log_message "Model: ${MODEL}"
log_message "Max jobs allowed: ${MAX_JOBS_ALLOWED}"
log_message "Check interval: ${CHECK_INTERVAL}s"
log_message "State file: ${STATE_FILE}"
log_message "============================================================================"

# ============================================================================
# MAIN MONITORING LOOP
# ============================================================================

declare -A SAMPLING_JOBS
declare -A ANALYSIS_JOBS_SUBMITTED

while true; do
    # Wait for capacity
    wait_for_capacity
    
    SUBMITTED_THIS_ROUND=0
    
    # Check each iteration
    for i in {1..5}; do
        STAGE=$(grep "^${i}," ${STATE_FILE} | cut -d',' -f2)
        STATUS=$(grep "^${i}," ${STATE_FILE} | cut -d',' -f4)
        
        # Submit sampling if pending
        if [[ "$STAGE" == "sampling_pending" ]]; then
            log_message "Iteration ${i}: Submitting sampling job..."
            
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

ID=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG_FILE})
SPLIT_INDEX=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG_FILE})
BENCHMARK_PATH=/scratch/project_465002574/UNAAGI_benchmarks/${ID}.pt

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

            # Replace placeholders
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
            
            SAMPLING_JOB=$(sbatch --parsable ${SAMPLING_SCRIPT} 2>&1)
            
            if [[ $SAMPLING_JOB =~ ^[0-9]+$ ]]; then
                SAMPLING_JOBS[$i]=${SAMPLING_JOB}
                sed -i "s|^${i},sampling_pending,,|${i},sampling_running,${SAMPLING_JOB},submitted|" ${STATE_FILE}
                log_message "  ✓ Iteration ${i} sampling submitted: ${SAMPLING_JOB}"
                SUBMITTED_THIS_ROUND=$((SUBMITTED_THIS_ROUND + 1))
                sleep 2
            else
                log_message "  ✗ Iteration ${i} sampling failed: ${SAMPLING_JOB}"
            fi
            
        # Check if sampling completed and submit analysis
        elif [[ "$STAGE" == "sampling_running" ]]; then
            JOB_ID=$(grep "^${i}," ${STATE_FILE} | cut -d',' -f3)
            
            # Check if all array tasks completed
            PENDING=$(squeue -j ${JOB_ID} -h -t pending,running 2>/dev/null | wc -l)
            
            if [ $PENDING -eq 0 ] && [ -z "${ANALYSIS_JOBS_SUBMITTED[$i]}" ]; then
                log_message "Iteration ${i}: Sampling completed, submitting analysis jobs..."
                
                # Submit 25 analysis jobs
                for protein_idx in {0..24}; do
                    ANALYSIS_SCRIPT="${SCRIPT_DIR}/analysis_iter${i}_protein${protein_idx}.sh"
                    
                    # Create analysis script (same as before, using heredoc)
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

module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

git fetch origin
git checkout main

MODEL=MODEL_PLACEHOLDER
CONFIG_FILE=CONFIG_FILE_PLACEHOLDER
NUM_SAMPLES=NUM_SAMPLES_PLACEHOLDER
TOTAL_NUM=TOTAL_NUM_PLACEHOLDER
PROTEIN_IDX=PROTEIN_IDX_PLACEHOLDER

ID=$(awk -v ArrayID=${PROTEIN_IDX} 'BEGIN{count=0} {if(count==ArrayID){print $2; exit} if(NR>1 && prev!=$2){count++} prev=$2}' ${CONFIG_FILE})
BASELINE=$(awk -v ID="${ID}" '$2==ID {print $3; exit}' ${CONFIG_FILE})

RUN_ID_BASE="${MODEL}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER"
SAMPLES_DIR="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID_BASE}_split0/Samples"
OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/${MODEL}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER"

echo "============================================================================"
echo "UAAG Analysis - Protein ${ID} (Iteration ITER)"
echo "============================================================================"

SAMPLES_PATH="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID_BASE}_split*/Samples"
python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}

python scripts/result_eval_uniform.py \
    --generated ${SAMPLES_DIR}/aa_distribution.csv \
    --baselines /scratch/project_465002574/UNAAGI_benchmark_values/baselines/${BASELINE} \
    --total_num ${TOTAL_NUM} \
    --output_dir ${OUTPUT_DIR}

ARCHIVE_DIR="/scratch/project_465002574/UNAAGI_archives/${MODEL}"
mkdir -p ${ARCHIVE_DIR}
ARCHIVE_NAME="${ARCHIVE_DIR}/${ID}_${MODEL}_variational_sampling_${NUM_SAMPLES}_ITER.tar.gz"

cd /scratch/project_465002574/ProteinGymSampling/
tar -czf ${ARCHIVE_NAME} run${RUN_ID_BASE}_split*/
rm -rf run${RUN_ID_BASE}_split*/

echo "[$(date)] Analysis completed: ${ARCHIVE_NAME}"
ANALYSIS_EOF

                    # Replace placeholders
                    sed -i "s|ITER|${i}|g" ${ANALYSIS_SCRIPT}
                    sed -i "s|PROT|${protein_idx}|g" ${ANALYSIS_SCRIPT}
                    sed -i "s|MODEL_PLACEHOLDER|${MODEL}|g" ${ANALYSIS_SCRIPT}
                    sed -i "s|CONFIG_FILE_PLACEHOLDER|${CONFIG_FILE}|g" ${ANALYSIS_SCRIPT}
                    sed -i "s|NUM_SAMPLES_PLACEHOLDER|${NUM_SAMPLES}|g" ${ANALYSIS_SCRIPT}
                    sed -i "s|TOTAL_NUM_PLACEHOLDER|${TOTAL_NUM}|g" ${ANALYSIS_SCRIPT}
                    sed -i "s|ANALYSIS_TIME_PLACEHOLDER|${ANALYSIS_TIME}|g" ${ANALYSIS_SCRIPT}
                    sed -i "s|ANALYSIS_PARTITION_PLACEHOLDER|${ANALYSIS_PARTITION}|g" ${ANALYSIS_SCRIPT}
                    sed -i "s|PROTEIN_IDX_PLACEHOLDER|${protein_idx}|g" ${ANALYSIS_SCRIPT}
                    
                    # Calculate dependencies
                    start_task=$((protein_idx * 10))
                    end_task=$((start_task + 9))
                    dep_string="afterok"
                    for task_id in $(seq ${start_task} ${end_task}); do
                        dep_string="${dep_string}:${JOB_ID}_${task_id}"
                    done
                    
                    sbatch --dependency=${dep_string} ${ANALYSIS_SCRIPT} > /dev/null 2>&1
                    sleep 0.5
                done
                
                ANALYSIS_JOBS_SUBMITTED[$i]=1
                sed -i "s|^${i},sampling_running,${JOB_ID},.*|${i},analysis_running,${JOB_ID},completed|" ${STATE_FILE}
                log_message "  ✓ Iteration ${i}: 25 analysis jobs submitted"
                SUBMITTED_THIS_ROUND=$((SUBMITTED_THIS_ROUND + 25))
            fi
        fi
    done
    
    # Submit CP2 and PUMA if not done yet
    if ! grep -q "^cp2,.*,completed" ${STATE_FILE}; then
        CP2_STATUS=$(grep "^cp2," ${STATE_FILE} | cut -d',' -f2)
        if [[ "$CP2_STATUS" == "sampling_pending" ]]; then
            log_message "Submitting CP2 benchmark..."
            CP2_JOB=$(sbatch --parsable run_sampling_CP2_sbatch.sh 2>&1)
            if [[ $CP2_JOB =~ ^[0-9]+$ ]]; then
                sed -i "s|^cp2,.*|cp2,running,${CP2_JOB},completed|" ${STATE_FILE}
                log_message "  ✓ CP2 submitted: ${CP2_JOB}"
            fi
        fi
    fi
    
    if ! grep -q "^puma,.*,completed" ${STATE_FILE}; then
        PUMA_STATUS=$(grep "^puma," ${STATE_FILE} | cut -d',' -f2)
        if [[ "$PUMA_STATUS" == "sampling_pending" ]]; then
            log_message "Submitting PUMA benchmark..."
            PUMA_JOB=$(sbatch --parsable run_sampling_PUMA_sbatch.sh 2>&1)
            if [[ $PUMA_JOB =~ ^[0-9]+$ ]]; then
                sed -i "s|^puma,.*|puma,running,${PUMA_JOB},completed|" ${STATE_FILE}
                log_message "  ✓ PUMA submitted: ${PUMA_JOB}"
            fi
        fi
    fi
    
    # Check if all done
    if grep -q "^5,analysis_running" ${STATE_FILE} && \
       grep -q "^cp2,running" ${STATE_FILE} && \
       grep -q "^puma,running" ${STATE_FILE}; then
        log_message "============================================================================"
        log_message "All jobs submitted! Monitoring will continue until completion."
        log_message "Press Ctrl+C to stop monitoring (jobs will continue running)"
        log_message "============================================================================"
        break
    fi
    
    if [ $SUBMITTED_THIS_ROUND -eq 0 ]; then
        log_message "No new jobs submitted this round, waiting ${CHECK_INTERVAL}s..."
        sleep ${CHECK_INTERVAL}
    else
        log_message "Submitted ${SUBMITTED_THIS_ROUND} jobs this round"
        sleep 10  # Brief pause before next check
    fi
done

log_message "Pipeline submission complete. Check logs/ for job status."
