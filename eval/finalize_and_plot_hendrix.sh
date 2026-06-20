#!/bin/bash
# ============================================================================
# finalize_and_plot_hendrix.sh — Hendrix (CUDA) port of finalize_and_plot.sh.
# Stages 3.5/4/5: CP2/PUMA per-iter scoring -> <TAG>_modeldata.csv -> plot group.
# Runs AFTER sampling (submitted with afterok by run_pipeline_hendrix.sh).
#
# Do not sbatch directly — use eval/run_pipeline_hendrix.sh.
# Required env: MODELS (space-sep tags), PIPE_SRC (repo checkout holding the new
#   eval/ + scripts/ code; passed by run_pipeline_hendrix.sh because $0 inside an
#   sbatch job points at the spool copy, not the repo).
# Optional: NUM_SAMPLES, RESULT_ROOT, SAMPLE_BASE, BENCHMARK_DIR, BASELINE_DIR,
#   UAA_CSV, MODELDATA_DIR, FIG_DIR.
# ============================================================================
#SBATCH --job-name=finalize
#SBATCH --account=boomsma
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl
#SBATCH --output=/datasets/biochem/unaagi/logs/finalize_%x_%j.log
#SBATCH --error=/datasets/biochem/unaagi/logs/finalize_%x_%j.log
set -euo pipefail

: "${MODELS:?set MODELS (space-separated model tags)}"
: "${PIPE_SRC:?set PIPE_SRC (repo checkout with eval/ + scripts/)}"
NUM_SAMPLES=${NUM_SAMPLES:-1000}
RESULT_ROOT=${RESULT_ROOT:-/datasets/biochem/unaagi/results}
SAMPLE_BASE=${SAMPLE_BASE:-/datasets/biochem/unaagi/ProteinGymSampling}
BENCHMARK_DIR=${BENCHMARK_DIR:-/home/qcx679/hantang/UAAG2/data/full_graph/benchmarks}
BASELINE_DIR=${BASELINE_DIR:-/home/qcx679/hantang/UAAG2/data/baselines}
UAA_CSV=${UAA_CSV:-/home/qcx679/hantang/UAAG2/data/uaa_benchmark_csv}
MODELDATA_DIR=${MODELDATA_DIR:-/datasets/biochem/unaagi/results/modeldata}
FIG_DIR=${FIG_DIR:-/datasets/biochem/unaagi/results/figures/pipeline_$(date +%Y%m%d_%H%M)}
N_ITERS=5

source /opt/software/mamba/1.4.1/etc/profile.d/conda.sh
conda activate targetdiff
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # GLIBCXX for scipy/pandas
export PYTHONPATH="${PIPE_SRC}/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
mkdir -p /datasets/biochem/unaagi/logs "$MODELDATA_DIR" "$FIG_DIR"

for TAG in $MODELS; do
    RESULT_BASE="${RESULT_ROOT}/${TAG}"
    echo "=== finalize ${TAG} ==="

    # --- CP2 / PUMA: per-iter UAA scoring -> raw csv in make_modeldata layout ---
    for ASSAY in CP2 PUMA; do
        BENCH="${UAA_CSV}/${ASSAY}_reframe.csv"
        [[ -f "$BENCH" ]] || { echo "[warn] missing UAA benchmark $BENCH (set UAA_CSV)"; continue; }
        for it in $(seq 0 $((N_ITERS-1))); do
            AA_CSV="${SAMPLE_BASE}/run${TAG}/${ASSAY}_${TAG}_${NUM_SAMPLES}_iter${it}/aa_distribution_split0.csv"
            [[ -f "$AA_CSV" ]] || { echo "[warn] missing $AA_CSV"; continue; }
            DEST="${RESULT_BASE}/${ASSAY}_${TAG}_iter${it}"; mkdir -p "$DEST"
            WORK="${DEST}/_scorework"; mkdir -p "$WORK"          # scorer forces results/<dir>/
            ( cd "$WORK" && python "${PIPE_SRC}/scripts/result_eval_uniform_uaa.py" \
                --benchmark "$BENCH" --aa_output "$AA_CSV" \
                --output_dir "out" --total_num "$NUM_SAMPLES" ) \
                || { echo "[warn] uaa score failed ${ASSAY} iter${it}"; continue; }
            cp "${WORK}/results/out/all_benchmark_results_raw.csv" \
               "${DEST}/all_benchmark_results_raw.csv"
        done
    done

    # --- ProteinGym: ensure spearman_per_iter.csv exists (sampling makes it inline) ---
    for P in A0A247D711_LISMN AICDA_HUMAN ARGR_ECOLI B2L11_HUMAN CCDB_ECOLI DLG4_RAT \
             DN7A_SACS2 ENVZ_ECOLI ENV_HV1B9 ERBB2_HUMAN FKBP3_HUMAN HCP_LAMBD IF1_ECOLI \
             ILF3_HUMAN OTU7A_HUMAN PKN1_HUMAN RS15_GEOSE SBI_STAAM SCIN_STAAR SOX30_HUMAN \
             SQSTM_MOUSE SUMO1_HUMAN TAT_HV1BR VG08_BPP22 VRPI_BPT7; do
        RD="${RESULT_BASE}/${P}_${TAG}_${NUM_SAMPLES}"
        if [[ ! -f "${RD}/spearman_per_iter.csv" ]]; then
            BCSV=$(ls ${BASELINE_DIR}/${P}_*.csv 2>/dev/null | sort | head -1) || true
            [[ -n "${BCSV:-}" ]] || { echo "[warn] no baseline for $P"; continue; }
            mkdir -p "$RD"
            python "${PIPE_SRC}/scripts/aggregate_splits.py" \
                --glob "${SAMPLE_BASE}/run${TAG}/${P}_${TAG}_${NUM_SAMPLES}_iter*/aa_distribution_split0.csv" \
                --output-dir "$RD" --baselines "$BCSV" --total-num "$NUM_SAMPLES" || true
        fi
    done

    # --- build modeldata.csv (+ collate baselines_pg) ---
    python "${PIPE_SRC}/eval/make_modeldata.py" \
        --result-base "$RESULT_BASE" --model-tag "$TAG" \
        --num-samples "$NUM_SAMPLES" --out-dir "$MODELDATA_DIR"
done

# --- stage 5: plot the assigned group ---
echo "=== plotting group: $MODELS ==="
python "${PIPE_SRC}/scripts/compare_models.py" --data-dir "$MODELDATA_DIR" --output-dir "$FIG_DIR"
echo "[done] figures -> $FIG_DIR"
ls -1 "$FIG_DIR"/*.svg
