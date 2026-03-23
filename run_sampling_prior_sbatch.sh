#!/bin/bash
#SBATCH --job-name=prior_sampling
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9
#SBATCH -o /scratch/project_465002574/UAAG_logs/prior_job_%A_%a.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/prior_job_%A_%a.log

echo "============================================================================"
echo "UAAG Prior (Free) Sampling"
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
CKPT_PATH=/flash/project_465002574/UAAG2_main/${MODEL}/last.ckpt

# Benchmark source is arbitrary for prior mode; keep a valid test graph .pt path.
BENCHMARK_PATH=/scratch/project_465002574/UNAAGI_benchmarks/5ly1_cp2.pt

SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}
PRIOR_SEED=$((42 + SLURM_ARRAY_TASK_ID))
NUM_SAMPLES=1000
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15

# Construct unique run ID with split index
RUN_ID="${MODEL}/PRIOR_FREE_${MODEL}_${NUM_SAMPLES}_split${SPLIT_INDEX}"

echo "[$(date)] Starting prior sampling for ${RUN_ID}..."
echo "Split index: ${SPLIT_INDEX}"
echo "Benchmark: ${BENCHMARK_PATH}"
echo "Prior seed: ${PRIOR_SEED}"

python scripts/generate_ligand.py \
    --load-ckpt ${CKPT_PATH} \
    --id ${RUN_ID} \
    --batch-size ${BATCH_SIZE} \
    --virtual_node_size ${VIRTUAL_NODE_SIZE} \
    --num-samples ${NUM_SAMPLES} \
    --benchmark-path ${BENCHMARK_PATH} \
    --split_index ${SPLIT_INDEX} \
    --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl \
    --prior-sampling \
    --prior-seed ${PRIOR_SEED}

if [ $? -eq 0 ]; then
    echo "[$(date)] Prior sampling completed for ${RUN_ID}"
else
    echo "[$(date)] Prior sampling failed with exit code $?"
    exit 1
fi

# Post-processing and analysis
SAMPLES_PATH="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID}/Samples_prior"
OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/${MODEL}/PRIOR_FREE_${MODEL}_${NUM_SAMPLES}_split${SPLIT_INDEX}"
ARCHIVE_DIR="/scratch/project_465002574/UNAAGI_archives/${MODEL}"

mkdir -p ${ARCHIVE_DIR}

echo ""
echo "============================================================================"
echo "Starting Prior Post-Processing and Analysis"
echo "============================================================================"
echo "[$(date)] Starting prior analysis for ${RUN_ID}..."

python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}

if [ $? -eq 0 ]; then
    echo "[$(date)] Post-processing completed successfully"
else
    echo "[$(date)] Post-processing failed with exit code $?"
    exit 1
fi

python scripts/result_eval_uniform_uaa.py \
    --benchmark /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/CP2_reframe.csv \
    --aa_output ${SAMPLES_PATH}/aa_distribution.csv \
    --output_dir ${OUTPUT_DIR} \
    --total_num ${NUM_SAMPLES}

if [ $? -eq 0 ]; then
    echo "[$(date)] Prior evaluation completed successfully"
else
    echo "[$(date)] Prior evaluation failed with exit code $?"
    exit 1
fi

# PoseBusters evaluation
POSEBUSTER_OUTPUT="${SAMPLES_PATH}/PoseBusterResults"
POSEBUSTER_TEMP_DIR="/flash/project_465002574/temp_sdf_prior_split${SPLIT_INDEX}"

echo ""
echo "============================================================================"
echo "Starting PoseBusters Evaluation"
echo "============================================================================"
echo "[$(date)] Evaluating generated structures with PoseBusters..."

python scripts/evaluate_mol_samples.py \
    --input-dir ${SAMPLES_PATH} \
    --output ${POSEBUSTER_OUTPUT} \
    --max-workers 6 \
    --temp-dir ${POSEBUSTER_TEMP_DIR}

if [ $? -eq 0 ]; then
    if [ -f "${POSEBUSTER_OUTPUT}" ]; then
        EVAL_LINES=$(wc -l < "${POSEBUSTER_OUTPUT}")
        echo "[$(date)] PoseBusters completed: $((EVAL_LINES - 1)) molecules evaluated"
    else
        echo "[$(date)] PoseBusters completed, output file not found at ${POSEBUSTER_OUTPUT}"
    fi
else
    echo "[$(date)] PoseBusters evaluation failed with exit code $?"
    exit 1
fi

# Compress molecule files
ARCHIVE_PATH="${ARCHIVE_DIR}/PRIOR_FREE_${MODEL}_${NUM_SAMPLES}_split${SPLIT_INDEX}_mol_files.tar.gz"

echo ""
echo "============================================================================"
echo "Compressing Molecule Files"
echo "============================================================================"
echo "[$(date)] Creating archive: ${ARCHIVE_PATH}"

cd ${SAMPLES_PATH}
MOL_COUNT=$(find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -type f | wc -l)

if [ ${MOL_COUNT} -gt 0 ]; then
    TAR_EXIT_CODE=0
    if tar --help 2>/dev/null | grep -q -- "--checkpoint-action"; then
        find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -type f -print0 | \
            tar --null -T - -czf "${ARCHIVE_PATH}" \
                --checkpoint=2000 \
                --checkpoint-action=echo='[tar] checkpoints processed: %u'
        TAR_EXIT_CODE=$?
    else
        find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -type f -print0 | \
            tar -czf "${ARCHIVE_PATH}" --null -T -
        TAR_EXIT_CODE=$?
    fi

    if [ ${TAR_EXIT_CODE} -eq 0 ]; then
        ARCHIVE_SIZE=$(du -h "${ARCHIVE_PATH}" | cut -f1)
        echo "[$(date)] Archive created successfully (${ARCHIVE_SIZE})"
    else
        echo "[$(date)] Failed to create archive, skipping cleanup for safety"
        exit 1
    fi
else
    echo "[$(date)] No .mol or all_molecules.sdf files found to archive"
fi

# Cleanup molecule files while keeping analysis outputs
echo ""
echo "============================================================================"
echo "Cleaning Up Raw Molecule Files"
echo "============================================================================"

MOL_BEFORE=$(find . -name "*.mol" -type f | wc -l)
SDF_BEFORE=$(find . -name "all_molecules.sdf" -type f | wc -l)
echo "[$(date)] Deleting $((MOL_BEFORE + SDF_BEFORE)) molecule files (${MOL_BEFORE} .mol + ${SDF_BEFORE} .sdf)"

find . -name "*.mol" -type f -delete
find . -name "all_molecules.sdf" -type f -delete
find . -type d -name "batch_*" -exec rm -rf {} + 2>/dev/null
find . -type d -name "iter_*" -exec rm -rf {} + 2>/dev/null
find . -type d -name "final" -exec rm -rf {} + 2>/dev/null

REMAINING_MOL=$(find . -name "*.mol" -type f | wc -l)
REMAINING_SDF=$(find . -name "all_molecules.sdf" -type f | wc -l)
DISK_USAGE=$(du -sh ${SAMPLES_PATH} | cut -f1)

echo "[$(date)] Cleanup complete"
echo "Remaining molecule files: $((REMAINING_MOL + REMAINING_SDF))"
echo "Samples directory size: ${DISK_USAGE}"

cd - >/dev/null

echo ""
echo "============================================================================"
echo "Prior Sampling + Analysis Completed Successfully"
echo "============================================================================"
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "PoseBusters output: ${POSEBUSTER_OUTPUT}"
echo "Archive saved to: ${ARCHIVE_PATH}"
echo "============================================================================"
