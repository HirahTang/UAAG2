#!/bin/bash
#SBATCH --job-name=PUMA_sampling
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu,boomsma
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9
#SBATCH --exclude=hendrixgpu01fl,hendrixgpu16fl,hendrixgpu19fl,hendrixgpu04fl,hendrixgpu26fl,hendrixgpu24fl,hendrixgpu25fl,hendrixgpu06fl
#SBATCH -o logs/PUMA_job_%A_%a.log
#SBATCH -e logs/PUMA_job_%A_%a.log

nvidia-smi
echo "Job $SLURM_JOB_ID is running on node: $SLURMD_NODENAME"
echo "Hostname: $(hostname)"

source ~/.bashrc
conda activate targetdiff
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qcx679/.conda/envs/targetdiff/lib

git fetch origin
git checkout main
echo "Running on branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit hash:       $(git rev-parse HEAD)"

MODEL=Full_mask_5_virtual_node_mask_token_atomic_only_mask_diffusion_0917
CKPT_PATH=/home/qcx679/hantang/UAAG2/3DcoordsAtomsBonds_0/run${MODEL}/last.ckpt
BENCHMARK_PATH=/home/qcx679/hantang/UAAG2/data/full_graph/benchmarks/2roc_puma.pt
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}
NUM_SAMPLES=1000

# Construct unique run ID with split index
RUN_ID="${MODEL}/PUMA_${MODEL}_variational_sampling_${NUM_SAMPLES}_split${SPLIT_INDEX}"

echo "[$(date)] Starting PUMA sampling for ${RUN_ID}..."
echo "Split index: ${SPLIT_INDEX}"
echo "Benchmark: ${BENCHMARK_PATH}"

python scripts/generate_ligand.py \
    --load-ckpt ${CKPT_PATH} \
    --id ${RUN_ID} \
    --batch-size 8 \
    --virtual_node_size 15 \
    --num-samples ${NUM_SAMPLES} \
    --benchmark-path ${BENCHMARK_PATH} \
    --split_index ${SPLIT_INDEX}

echo "[$(date)] PUMA sampling completed for ${RUN_ID}"

# Post-processing and analysis
SAMPLES_PATH="/datasets/biochem/unaagi/ProteinGymSampling/run${RUN_ID}/Samples"
OUTPUT_DIR="/home/qcx679/hantang/UAAG2/results/${MODEL}/PUMA_${MODEL}_variational_sampling_${NUM_SAMPLES}_split${SPLIT_INDEX}"

echo "[$(date)] Starting PUMA analysis for ${RUN_ID}..."
python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}
python scripts/result_eval_uniform_uaa.py \
    --benchmark /home/qcx679/hantang/UAAG2/data/uaa_benchmark_csv/PUMA_reframe.csv \
    --aa_output ${SAMPLES_PATH}/aa_distribution.csv \
    --output_dir ${OUTPUT_DIR} \
    --total_num ${NUM_SAMPLES}

echo "[$(date)] PUMA analysis completed for ${RUN_ID}"
echo "Results saved to: ${OUTPUT_DIR}"
