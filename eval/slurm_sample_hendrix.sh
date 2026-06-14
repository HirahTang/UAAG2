#!/bin/bash
# Hendrix port of eval/slurm_sample.sh — CTMC + 20 NFE, 1000 samples x 5 iters
# over 27 proteins (25 ProteinGym + CP2 + PUMA). CUDA / targetdiff conda env.
#
# Do not sbatch directly — use:  eval/run_eval_hendrix.sh <model_name>
# Required env: CKPT, MODEL_TAG, SRCDIR.  Optional: RESULT_BASE, NUM_SAMPLES.
#
# Differences vs LUMI version: conda targetdiff (no LUMI modules/ROCm),
# --account=boomsma --partition=gpu --gres=gpu:1, exclude known-bad GPU nodes,
# Hendrix data paths, and --data_info_path override (checkpoint's baked-in path
# is a LUMI /flash path that does not exist here).

#SBATCH --job-name=uaag_eval
#SBATCH --account=boomsma
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl
#SBATCH --array=0-134%20
#SBATCH --output=/datasets/biochem/unaagi/logs/eval_%x_%A_%a.log
#SBATCH --error=/datasets/biochem/unaagi/logs/eval_%x_%A_%a.log

set -u
: "${CKPT:?set CKPT}"; : "${MODEL_TAG:?set MODEL_TAG}"; : "${SRCDIR:?set SRCDIR}"
NUM_SAMPLES=${NUM_SAMPLES:-1000}
DATA_INFO=/home/qcx679/hantang/UAAG2/data/statistic.pkl
BENCHMARK_DIR=/home/qcx679/hantang/UAAG2/data/full_graph/benchmarks
BASELINE_DIR=/home/qcx679/hantang/UAAG2/data/baselines
SAMPLE_BASE=/datasets/biochem/unaagi/ProteinGymSampling
RESULT_BASE=${RESULT_BASE:-/datasets/biochem/unaagi/results/${MODEL_TAG}}
mkdir -p /datasets/biochem/unaagi/logs "$RESULT_BASE"

PROTEINS=(
    A0A247D711_LISMN AICDA_HUMAN ARGR_ECOLI B2L11_HUMAN CCDB_ECOLI DLG4_RAT
    DN7A_SACS2 ENVZ_ECOLI ENV_HV1B9 ERBB2_HUMAN FKBP3_HUMAN HCP_LAMBD IF1_ECOLI
    ILF3_HUMAN OTU7A_HUMAN PKN1_HUMAN RS15_GEOSE SBI_STAAM SCIN_STAAR SOX30_HUMAN
    SQSTM_MOUSE SUMO1_HUMAN TAT_HV1BR VG08_BPP22 VRPI_BPT7 CP2 PUMA
)
N_ITERS=5
PROTEIN=${PROTEINS[$(( SLURM_ARRAY_TASK_ID / N_ITERS ))]}
ITER=$(( SLURM_ARRAY_TASK_ID % N_ITERS ))
echo "=== ${MODEL_TAG} | task ${SLURM_ARRAY_TASK_ID}: ${PROTEIN} iter${ITER} | node $(hostname) | SRCDIR=${SRCDIR} ==="

if   [[ "$PROTEIN" == "CP2"  ]]; then BENCH_PT=${BENCHMARK_DIR}/5ly1_cp2.pt;  IS_UAA=true
elif [[ "$PROTEIN" == "PUMA" ]]; then BENCH_PT=${BENCHMARK_DIR}/2roc_puma.pt; IS_UAA=true
else BENCH_PT=${BENCHMARK_DIR}/${PROTEIN}.pt; IS_UAA=false; fi

source /opt/software/mamba/1.4.1/etc/profile.d/conda.sh
conda activate targetdiff
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # fixes GLIBCXX for scipy
export PYTHONPATH="${SRCDIR}/src:${PYTHONPATH:-}"

echo "[$(date)] sampling ${PROTEIN} iter${ITER}"
python ${SRCDIR}/scripts/generate_ligand_dpm.py \
    --load-ckpt   "$CKPT" \
    --id          "${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter${ITER}" \
    --benchmark-path "$BENCH_PT" \
    --save-dir    "$SAMPLE_BASE" \
    --data_info_path "$DATA_INFO" \
    --num-samples "$NUM_SAMPLES" \
    --split_index 0 --total_partition 1 \
    --every-k-step 25 --ctmc --dpm-solver-pp --batch-size 8 || { echo "[ERROR] sampling failed"; exit 1; }
echo "[$(date)] sampling done."

if [[ "$IS_UAA" == "false" ]]; then
    BASELINE_CSV=$(ls ${BASELINE_DIR}/${PROTEIN}_*.csv 2>/dev/null | sort | head -1)
    if [[ -n "$BASELINE_CSV" ]]; then
        RESULT_DIR=${RESULT_BASE}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}
        mkdir -p "$RESULT_DIR"
        python ${SRCDIR}/scripts/aggregate_splits.py \
            --glob       "${SAMPLE_BASE}/run${MODEL_TAG}/${PROTEIN}_${MODEL_TAG}_${NUM_SAMPLES}_iter*/aa_distribution_split0.csv" \
            --output-dir "$RESULT_DIR" \
            --baselines  "$BASELINE_CSV" \
            --total-num  "$NUM_SAMPLES" && echo "[$(date)] aggregated -> ${RESULT_DIR}" || echo "[WARN] aggregate failed (score later via result_eval_uniform.py)"
    fi
fi
