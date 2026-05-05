#!/bin/bash -l
# Re-run ctmc_20nfe sampling (100 samples, 1 iter) and evaluate through
# aggregate_splits.py so scores are computed with the same method as ctmc_1k_5iter.
#
# Usage:
#   sbatch slurm/submit_ctmc_20nfe_eval.sh
#
# Output:
#   Sampling:    /scratch/project_465002574/ProteinGymSampling/runctmc_20nfe_eval/
#   Aggregated:  /scratch/project_465002574/UNAAGI_result/results/ctmc_20nfe_eval/

#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctmc_20nfe_eval
#SBATCH --array=0-23
#SBATCH --output=/scratch/project_465002574/uaag2_adit_results/logs/ctmc_20nfe_eval_%A_%a.out
#SBATCH --error=/scratch/project_465002574/uaag2_adit_results/logs/ctmc_20nfe_eval_%A_%a.err

# ---------------------------------------------------------------------------
# Benchmark list (24 assays — same as original ctmc_20nfe run)
# ---------------------------------------------------------------------------
PROTEINS=(
    A0A247D711_LISMN
    AICDA_HUMAN
    ARGR_ECOLI
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
)

PROTEIN=${PROTEINS[$SLURM_ARRAY_TASK_ID]}
echo "=== Array task ${SLURM_ARRAY_TASK_ID}: ${PROTEIN} ==="

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_40_gpu_UAAG_model_official_8_0202/last.ckpt
SRCDIR=/flash/project_465002574/UAAG2_main
BENCHMARK_DIR=/scratch/project_465002574/UNAAGI_benchmarks
BASELINE_DIR=/scratch/project_465002574/UNAAGI_benchmark_values/baselines
SAMPLE_BASE=/scratch/project_465002574/ProteinGymSampling/runctmc_20nfe_eval
RESULT_BASE=/scratch/project_465002574/UNAAGI_result/results/ctmc_20nfe_eval

MODEL_TAG=ctmc_20nfe_eval
NUM_SAMPLES=100

# Resolve baseline CSV (alphabetical → picks Adkar over Tripathi for CCDB)
BASELINE_CSV=$(ls ${BASELINE_DIR}/${PROTEIN}_*.csv 2>/dev/null | sort | head -1)
if [[ -z "$BASELINE_CSV" ]]; then
    echo "[ERROR] No baseline CSV found for ${PROTEIN}" >&2
    exit 1
fi
echo "Baseline: ${BASELINE_CSV}"

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
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"

# ---------------------------------------------------------------------------
# Step 1: Sampling
# ---------------------------------------------------------------------------
SAMPLE_DIR=${SAMPLE_BASE}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter0
mkdir -p "$SAMPLE_DIR"
mkdir -p /scratch/project_465002574/uaag2_adit_results/logs

echo "[$(date)] Starting sampling for ${PROTEIN} ..."
python ${SRCDIR}/scripts/generate_ligand_dpm.py \
    --load-ckpt   "$CKPT" \
    --id          "${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter0" \
    --benchmark-path "${BENCHMARK_DIR}/${PROTEIN}.pt" \
    --save-dir    "$SAMPLE_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --split_index 0 --total_partition 1 \
    --every-k-step 25 \
    --ctmc \
    --dpm-solver-pp \
    --batch-size 8

SAMPLE_EXIT=$?
if [[ $SAMPLE_EXIT -ne 0 ]]; then
    echo "[ERROR] Sampling failed (exit ${SAMPLE_EXIT})" >&2
    exit $SAMPLE_EXIT
fi
echo "[$(date)] Sampling done."

# ---------------------------------------------------------------------------
# Step 2: Aggregate → spearman_per_iter.csv
# ---------------------------------------------------------------------------
RESULT_DIR=${RESULT_BASE}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}
mkdir -p "$RESULT_DIR"

echo "[$(date)] Running aggregate_splits for ${PROTEIN} ..."
python ${SRCDIR}/scripts/aggregate_splits.py \
    --glob        "${SAMPLE_DIR}/aa_distribution_split0.csv" \
    --output-dir  "$RESULT_DIR" \
    --baselines   "$BASELINE_CSV" \
    --total-num   "$NUM_SAMPLES"

AGG_EXIT=$?
if [[ $AGG_EXIT -ne 0 ]]; then
    echo "[ERROR] aggregate_splits failed (exit ${AGG_EXIT})" >&2
    exit $AGG_EXIT
fi

echo "[$(date)] Done. Results at: ${RESULT_DIR}"
echo "  spearman_per_iter.csv: $(cat ${RESULT_DIR}/spearman_per_iter.csv)"
