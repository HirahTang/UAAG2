#!/bin/bash -l
# Sample with new prior checkpoint using DDPM 500 timesteps, 300 samples × 3 iterations
# over 25 ProteinGym assays + CP2 + PUMA (27 proteins total).
#
# Array layout: protein_idx = task_id // 3, iter_idx = task_id % 3
# Total: 27 proteins × 3 iters = 81 tasks (array 0-80)
#
# Usage:
#   sbatch slurm/submit_prior_ddpm_300_3iter.sh
#
# Sampling output:  /scratch/project_465002574/ProteinGymSampling/runp0203_ddpm/
# Aggregated:       /scratch/project_465002574/UNAAGI_result/results/p0203_ddpm/

#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --job-name=prior_ddpm_300
#SBATCH --array=0-80
#SBATCH --output=/scratch/project_465002574/uaag2_adit_results/logs/prior_ddpm_300_%A_%a.out
#SBATCH --error=/scratch/project_465002574/uaag2_adit_results/logs/prior_ddpm_300_%A_%a.err

# ---------------------------------------------------------------------------
# Protein list (27 total: 25 ProteinGym + CP2 + PUMA)
# ---------------------------------------------------------------------------
PROTEINS=(
    A0A247D711_LISMN
    AICDA_HUMAN
    ARGR_ECOLI
    B2L11_HUMAN
    CCDB_ECOLI
    DLG4_RAT
    DN7A_SACS2
    ENVZ_ECOLI
    ENV_HV1B9
    ERBB2_HUMAN
    FKBP3_HUMAN
    HCP_LAMBD
    IF1_ECOLI
    ILF3_HUMAN
    OTU7A_HUMAN
    PKN1_HUMAN
    RS15_GEOSE
    SBI_STAAM
    SCIN_STAAR
    SOX30_HUMAN
    SQSTM_MOUSE
    SUMO1_HUMAN
    TAT_HV1BR
    VG08_BPP22
    VRPI_BPT7
    CP2
    PUMA
)

N_ITERS=3
PROTEIN_IDX=$(( SLURM_ARRAY_TASK_ID / N_ITERS ))
ITER=$(( SLURM_ARRAY_TASK_ID % N_ITERS ))
PROTEIN=${PROTEINS[$PROTEIN_IDX]}

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: ${PROTEIN} iter${ITER} ==="

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_8_gpu_UAAG_model_official_8_0203_prior/last-v1.ckpt
SRCDIR=/flash/project_465002574/UAAG2_main
BENCHMARK_DIR=/scratch/project_465002574/UNAAGI_benchmarks
BASELINE_DIR=/scratch/project_465002574/UNAAGI_benchmark_values/baselines
SAMPLE_BASE=/scratch/project_465002574/ProteinGymSampling
RESULT_BASE=/scratch/project_465002574/UNAAGI_result/results/p0203_ddpm_true

MODEL_TAG=p0203_ddpm
NUM_SAMPLES=300

# Benchmark .pt path (CP2 and PUMA have special filenames)
if [[ "$PROTEIN" == "CP2" ]]; then
    BENCH_PT=${BENCHMARK_DIR}/5ly1_cp2.pt
    IS_UAA=true
elif [[ "$PROTEIN" == "PUMA" ]]; then
    BENCH_PT=${BENCHMARK_DIR}/2roc_puma.pt
    IS_UAA=true
else
    BENCH_PT=${BENCHMARK_DIR}/${PROTEIN}.pt
    IS_UAA=false
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default

export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export PYTHONPATH="${SRCDIR}/src:$PYTHONPATH"
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p /scratch/project_465002574/uaag2_adit_results/logs

# ---------------------------------------------------------------------------
# Sampling (DDPM: no --ctmc, no --every-k-step, no --dpm-solver-pp)
# ---------------------------------------------------------------------------
# Output will land at: ${SAMPLE_BASE}/run${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter${ITER}/

echo "[$(date)] Starting sampling: ${PROTEIN} iter${ITER}"
python ${SRCDIR}/scripts/generate_ligand_dpm.py \
    --load-ckpt   "$CKPT" \
    --id          "${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter${ITER}" \
    --benchmark-path "$BENCH_PT" \
    --save-dir    "$SAMPLE_BASE" \
    --num-samples "$NUM_SAMPLES" \
    --split_index 0 --total_partition 1 --every-k-step 1 \
    --batch-size 8

SAMPLE_EXIT=$?
if [[ $SAMPLE_EXIT -ne 0 ]]; then
    echo "[ERROR] Sampling failed (exit ${SAMPLE_EXIT})" >&2
    exit $SAMPLE_EXIT
fi
echo "[$(date)] Sampling done."

# ---------------------------------------------------------------------------
# Aggregate (ProteinGym proteins only; CP2/PUMA scored separately by plot script)
# ---------------------------------------------------------------------------
if [[ "$IS_UAA" == "false" ]]; then
    BASELINE_CSV=$(ls ${BASELINE_DIR}/${PROTEIN}_*.csv 2>/dev/null | sort | head -1)
    if [[ -z "$BASELINE_CSV" ]]; then
        echo "[WARN] No baseline CSV for ${PROTEIN}, skipping aggregation"
    else
        RESULT_DIR=${RESULT_BASE}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}
        mkdir -p "$RESULT_DIR"
        echo "[$(date)] Running aggregate_splits for ${PROTEIN} ..."
        python ${SRCDIR}/scripts/aggregate_splits.py \
            --glob       "${SAMPLE_BASE}/run${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter*/aa_distribution_split0.csv" \
            --output-dir "$RESULT_DIR" \
            --baselines  "$BASELINE_CSV" \
            --total-num  "$NUM_SAMPLES"
        echo "[$(date)] Aggregation done. Results at: ${RESULT_DIR}"
    fi
fi
