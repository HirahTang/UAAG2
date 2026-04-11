#!/bin/bash
#SBATCH --job-name=prior_sampling_PUMA
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9
#SBATCH -o /scratch/project_465002574/UAAG_logs/prior_puma_%A_%a.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/prior_puma_%A_%a.log

echo "============================================================================"
echo "UAAG Prior (Free) Sampling — PUMA"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID, Array task: $SLURM_ARRAY_TASK_ID, Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

cd /flash/project_465002574/UAAG2_main

MODEL=runFull_mask_8_gpu_UAAG_model_official_8_0203_prior
CKPT_PATH=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/${MODEL}/last-v1.ckpt
BENCHMARK_PATH=/scratch/project_465002574/UNAAGI_benchmarks/2roc_puma.pt
BENCHMARK_CSV=/scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/PUMA_reframe.csv

SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}
PRIOR_SEED=$((42 + SLURM_ARRAY_TASK_ID))
NUM_SAMPLES=1000
BATCH_SIZE=8
VIRTUAL_NODE_SIZE=15

RUN_ID="${MODEL}/PUMA_PRIOR_FREE_${MODEL}_${NUM_SAMPLES}_split${SPLIT_INDEX}"
SAMPLES_PATH="/scratch/project_465002574/ProteinGymSampling/run${RUN_ID}/Samples_prior"
OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/${MODEL}/PUMA_PRIOR_FREE_${MODEL}_${NUM_SAMPLES}_split${SPLIT_INDEX}"
ARCHIVE_DIR="/scratch/project_465002574/UNAAGI_archives/${MODEL}"

echo "[$(date)] Starting prior sampling for ${RUN_ID}..."

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

python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}

mkdir -p ${OUTPUT_DIR}
python scripts/result_eval_uniform_uaa.py \
    --benchmark ${BENCHMARK_CSV} \
    --aa_output ${SAMPLES_PATH}/aa_distribution.csv \
    --output_dir ${OUTPUT_DIR} \
    --total_num ${NUM_SAMPLES}

# Compress and clean up
mkdir -p ${ARCHIVE_DIR}
ARCHIVE_PATH="${ARCHIVE_DIR}/PUMA_PRIOR_FREE_${MODEL}_${NUM_SAMPLES}_split${SPLIT_INDEX}_mol_files.tar.gz"
cd ${SAMPLES_PATH}
find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -type f -print0 | \
    tar -czf "${ARCHIVE_PATH}" --null -T - 2>/dev/null || true
find . -name "*.mol" -type f -delete
find . -name "all_molecules.sdf" -type f -delete
find . -type d \( -name "batch_*" -o -name "iter_*" -o -name "final" \) -exec rm -rf {} + 2>/dev/null || true

echo "[$(date)] Done. Results: ${OUTPUT_DIR}"
