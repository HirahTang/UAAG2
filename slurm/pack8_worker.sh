#!/bin/bash
# One sampling+aggregation unit of work, pinned to a single GCD.
#
# Reads TASK_ID (0..N_TASKS-1) and derives (protein, iteration) exactly as the
# original one-task-per-job array did, so results are byte-for-byte comparable.
# Env expected: CKPT SRCDIR MODEL_TAG SAMPLE_BASE RESULT_BASE BENCHMARK_DIR
#               BASELINE_DIR NUM_SAMPLES EXTRA_ARGS TASK_ID
set -u

PROTEINS=(
    A0A247D711_LISMN AICDA_HUMAN ARGR_ECOLI B2L11_HUMAN CCDB_ECOLI DLG4_RAT
    DN7A_SACS2 ENVZ_ECOLI ENV_HV1B9 ERBB2_HUMAN FKBP3_HUMAN HCP_LAMBD IF1_ECOLI
    ILF3_HUMAN OTU7A_HUMAN PKN1_HUMAN RS15_GEOSE SBI_STAAM SCIN_STAAR SOX30_HUMAN
    SQSTM_MOUSE SUMO1_HUMAN TAT_HV1BR VG08_BPP22 VRPI_BPT7 CP2 PUMA
)
N_ITERS=5
N_TASKS=$(( ${#PROTEINS[@]} * N_ITERS ))

if [[ "$TASK_ID" -ge "$N_TASKS" ]]; then
    echo "[rank $SLURM_PROCID] task $TASK_ID >= $N_TASKS — no work, exiting cleanly"
    exit 0
fi

PROTEIN=${PROTEINS[$(( TASK_ID / N_ITERS ))]}
ITER=$(( TASK_ID % N_ITERS ))

if [[ "$PROTEIN" == "CP2" ]]; then
    BENCH_PT=${BENCHMARK_DIR}/5ly1_cp2.pt; IS_UAA=true
elif [[ "$PROTEIN" == "PUMA" ]]; then
    BENCH_PT=${BENCHMARK_DIR}/2roc_puma.pt; IS_UAA=true
else
    BENCH_PT=${BENCHMARK_DIR}/${PROTEIN}.pt; IS_UAA=false
fi

# Pin this rank to its own GCD and give it a private MIOpen cache, or the 8
# concurrent ranks corrupt each other's kernel database.
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}_${SLURM_PROCID}"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"

echo "[$(date)] rank ${SLURM_PROCID} gcd ${SLURM_LOCALID}: task ${TASK_ID} = ${PROTEIN} iter${ITER}"

python ${SRCDIR}/scripts/generate_ligand_dpm.py \
    --load-ckpt   "$CKPT" \
    --id          "${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter${ITER}" \
    --benchmark-path "$BENCH_PT" \
    --save-dir    "$SAMPLE_BASE" \
    --num-samples "$NUM_SAMPLES" \
    --split_index 0 --total_partition 1 \
    --every-k-step 25 \
    --ctmc \
    --dpm-solver-pp \
    --batch-size 8 \
    ${EXTRA_ARGS}
rc=$?
rm -rf "$MIOPEN_USER_DB_PATH"
if [[ $rc -ne 0 ]]; then
    echo "[ERROR] rank ${SLURM_PROCID} task ${TASK_ID} (${PROTEIN} iter${ITER}) failed: $rc" >&2
    exit $rc
fi
echo "[$(date)] rank ${SLURM_PROCID}: sampling done for ${PROTEIN} iter${ITER}"

if [[ "$IS_UAA" == "false" ]]; then
    BASELINE_CSV=$(ls ${BASELINE_DIR}/${PROTEIN}_*.csv 2>/dev/null | sort | head -1)
    if [[ -z "$BASELINE_CSV" ]]; then
        echo "[WARN] No baseline CSV for ${PROTEIN}, skipping aggregation"
    else
        RESULT_DIR=${RESULT_BASE}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}
        mkdir -p "$RESULT_DIR"
        python ${SRCDIR}/scripts/aggregate_splits.py \
            --glob       "${SAMPLE_BASE}/run${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter*/aa_distribution_split0.csv" \
            --output-dir "$RESULT_DIR" \
            --baselines  "$BASELINE_CSV" \
            --total-num  "$NUM_SAMPLES"
        echo "[$(date)] rank ${SLURM_PROCID}: aggregation done -> ${RESULT_DIR}"
    fi
fi
