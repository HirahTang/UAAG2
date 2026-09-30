#!/bin/bash -l
# Parametrized UNAAGI sampling array — CTMC + 20 NFE, 1000 samples x 5 iters
# over 27 proteins (25 ProteinGym + CP2 + PUMA). Generalizes
# slurm/submit_prior_ctmc_1k_5iter.sh: CKPT / MODEL_TAG / SRCDIR come from the
# environment (set by run_eval.sh via --export), so one script serves every model.
#
# Do not sbatch this directly — use:  eval/run_eval.sh <model_name>
# Required env: CKPT, MODEL_TAG, SRCDIR.  Optional: RESULT_BASE, NUM_SAMPLES.
#
# Array layout: protein_idx = task_id / 5, iter_idx = task_id % 5  (27 x 5 = 135)

#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-134%30
#SBATCH --output=/scratch/project_465002574/uaag2_adit_results/logs/eval_%x_%A_%a.out
#SBATCH --error=/scratch/project_465002574/uaag2_adit_results/logs/eval_%x_%A_%a.err

set -u
: "${CKPT:?set CKPT}"; : "${MODEL_TAG:?set MODEL_TAG}"; : "${SRCDIR:?set SRCDIR}"
NUM_SAMPLES=${NUM_SAMPLES:-1000}
RESULT_BASE=${RESULT_BASE:-/scratch/project_465002574/UNAAGI_result/results/${MODEL_TAG}}
BENCHMARK_DIR=/scratch/project_465002574/UNAAGI_benchmarks
BASELINE_DIR=/scratch/project_465002574/UNAAGI_benchmark_values/baselines
SAMPLE_BASE=/scratch/project_465002574/ProteinGymSampling

PROTEINS=(
    A0A247D711_LISMN AICDA_HUMAN ARGR_ECOLI B2L11_HUMAN CCDB_ECOLI DLG4_RAT
    DN7A_SACS2 ENVZ_ECOLI ENV_HV1B9 ERBB2_HUMAN FKBP3_HUMAN HCP_LAMBD IF1_ECOLI
    ILF3_HUMAN OTU7A_HUMAN PKN1_HUMAN RS15_GEOSE SBI_STAAM SCIN_STAAR SOX30_HUMAN
    SQSTM_MOUSE SUMO1_HUMAN TAT_HV1BR VG08_BPP22 VRPI_BPT7 CP2 PUMA
)
N_ITERS=5
PROTEIN=${PROTEINS[$(( SLURM_ARRAY_TASK_ID / N_ITERS ))]}
ITER=$(( SLURM_ARRAY_TASK_ID % N_ITERS ))
echo "=== ${MODEL_TAG} | task ${SLURM_ARRAY_TASK_ID}: ${PROTEIN} iter${ITER} | SRCDIR=${SRCDIR} ==="

# CP2/PUMA use special benchmark filenames and are scored separately (result_eval_uniform_uaa.py)
if   [[ "$PROTEIN" == "CP2"  ]]; then BENCH_PT=${BENCHMARK_DIR}/5ly1_cp2.pt;  IS_UAA=true
elif [[ "$PROTEIN" == "PUMA" ]]; then BENCH_PT=${BENCHMARK_DIR}/2roc_puma.pt; IS_UAA=true
else BENCH_PT=${BENCHMARK_DIR}/${PROTEIN}.pt; IS_UAA=false; fi

module purge; module load LUMI; module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export PYTHONPATH="${SRCDIR}/src:$PYTHONPATH"
export HSA_ENABLE_SDMA=0; export AMD_DIRECT_DISPATCH=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p /scratch/project_465002574/uaag2_adit_results/logs

echo "[$(date)] sampling ${PROTEIN} iter${ITER}"
python ${SRCDIR}/scripts/generate_ligand_dpm.py \
    --load-ckpt   "$CKPT" \
    --id          "${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter${ITER}" \
    --benchmark-path "$BENCH_PT" \
    --save-dir    "$SAMPLE_BASE" \
    --num-samples "$NUM_SAMPLES" \
    --split_index 0 --total_partition 1 \
    --every-k-step 25 --ctmc --dpm-solver-pp --batch-size 8 || { echo "[ERROR] sampling failed"; exit 1; }
echo "[$(date)] sampling done."

# Quick-look aggregation for ProteinGym proteins (CP2/PUMA via result_eval_uniform_uaa.py later)
if [[ "$IS_UAA" == "false" ]]; then
    BASELINE_CSV=$(ls ${BASELINE_DIR}/${PROTEIN}_*.csv 2>/dev/null | sort | head -1)
    if [[ -n "$BASELINE_CSV" ]]; then
        RESULT_DIR=${RESULT_BASE}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}
        mkdir -p "$RESULT_DIR"
        python ${SRCDIR}/scripts/aggregate_splits.py \
            --glob       "${SAMPLE_BASE}/run${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter*/aa_distribution_split0.csv" \
            --output-dir "$RESULT_DIR" \
            --baselines  "$BASELINE_CSV" \
            --total-num  "$NUM_SAMPLES" && echo "[$(date)] aggregated -> ${RESULT_DIR}"
    fi
fi
