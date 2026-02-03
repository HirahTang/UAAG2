#!/bin/bash
#SBATCH --job-name=CP2_sampling
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9
#SBATCH -o logs/CP2_job_%A_%a.log
#SBATCH -e logs/CP2_job_%A_%a.log

rocm-smi || echo "Warning: rocm-smi not available"
echo "Job $SLURM_JOB_ID is running on node: $SLURMD_NODENAME"
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

git fetch origin
git checkout main
echo "Running on branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit hash:       $(git rev-parse HEAD)"

MODEL=Full_mask_5_virtual_node_mask_token_atomic_only_mask_diffusion_0917
CKPT_PATH=/home/qcx679/hantang/UAAG2/3DcoordsAtomsBonds_0/run${MODEL}/last.ckpt
BENCHMARK_PATH=/home/qcx679/hantang/UAAG2/data/full_graph/benchmarks/5ly1_cp2.pt
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}
NUM_SAMPLES=1000

# Construct unique run ID with split index
RUN_ID="${MODEL}/CP2_${MODEL}_variational_sampling_${NUM_SAMPLES}_split${SPLIT_INDEX}"

echo "[$(date)] Starting CP2 sampling for ${RUN_ID}..."
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

echo "[$(date)] CP2 sampling completed for ${RUN_ID}"

# Post-processing and analysis
SAMPLES_PATH="/datasets/biochem/unaagi/ProteinGymSampling/run${RUN_ID}/Samples"
OUTPUT_DIR="/home/qcx679/hantang/UAAG2/results/${MODEL}/CP2_${MODEL}_variational_sampling_${NUM_SAMPLES}_split${SPLIT_INDEX}"

echo "[$(date)] Starting CP2 analysis for ${RUN_ID}..."
python scripts/post_analysis.py --analysis_path ${SAMPLES_PATH}
python scripts/result_eval_uniform_uaa.py \
    --benchmark /home/qcx679/hantang/UAAG2/data/uaa_benchmark_csv/CP2_reframe.csv \
    --aa_output ${SAMPLES_PATH}/aa_distribution.csv \
    --output_dir ${OUTPUT_DIR} \
    --total_num ${NUM_SAMPLES}

echo "[$(date)] CP2 analysis completed for ${RUN_ID}"
echo "Results saved to: ${OUTPUT_DIR}"