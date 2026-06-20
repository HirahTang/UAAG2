#!/bin/bash -l
# ============================================================================
# finalize_and_plot.sh — stages 3.5/4/5 of the one-button pipeline.
# Runs AFTER sampling (submitted with a SLURM afterok dependency by run_pipeline.sh).
#
# For each model in $MODELS:
#   - (CP2/PUMA) score each of the 5 iters with result_eval_uniform_uaa.py and
#     stash all_benchmark_results_raw.csv into the layout make_modeldata.py expects
#   - (ProteinGym) ensure each assay's spearman_per_iter.csv exists (re-aggregate if missing)
#   - build <TAG>_modeldata.csv via eval/make_modeldata.py
# Then plot the whole group with scripts/compare_models.py -> SVGs.
#
# Do not sbatch directly — use eval/run_pipeline.sh. Required env: MODELS, SRCDIR.
# Optional: NUM_SAMPLES, RESULT_ROOT, SAMPLE_BASE, MODELDATA_DIR, FIG_DIR.
# ============================================================================
#SBATCH --account=project_465002574
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465002574/uaag2_adit_results/logs/finalize_%x_%j.out
#SBATCH --error=/scratch/project_465002574/uaag2_adit_results/logs/finalize_%x_%j.err
set -euo pipefail

: "${MODELS:?set MODELS (space-separated model tags)}"
: "${SRCDIR:?set SRCDIR (repo checkout with scripts/)}"
NUM_SAMPLES=${NUM_SAMPLES:-1000}
RESULT_ROOT=${RESULT_ROOT:-/scratch/project_465002574/UNAAGI_result/results}
SAMPLE_BASE=${SAMPLE_BASE:-/scratch/project_465002574/ProteinGymSampling}
BASELINE_DIR=/scratch/project_465002574/UNAAGI_benchmark_values/baselines
UAA_CSV=/scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv
MODELDATA_DIR=${MODELDATA_DIR:-/scratch/project_465002574/UNAAGI_result/modeldata}
FIG_DIR=${FIG_DIR:-/scratch/project_465002574/UNAAGI_result/figures/pipeline_$(date +%Y%m%d_%H%M)}
N_ITERS=5

module purge; module load LUMI; module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
export PYTHONPATH="${SRCDIR}/src:$PYTHONPATH"
mkdir -p "$MODELDATA_DIR" "$FIG_DIR"

for TAG in $MODELS; do
    RESULT_BASE="${RESULT_ROOT}/${TAG}"
    echo "=== finalize ${TAG} ==="

    # --- CP2 / PUMA: per-iter UAA scoring -> raw csv in make_modeldata layout ---
    for ASSAY in CP2 PUMA; do
        BENCH="${UAA_CSV}/${ASSAY}_reframe.csv"
        [[ -f "$BENCH" ]] || { echo "[warn] missing UAA benchmark $BENCH"; continue; }
        for it in $(seq 0 $((N_ITERS-1))); do
            AA_CSV="${SAMPLE_BASE}/run${TAG}/${ASSAY}_${TAG}_${NUM_SAMPLES}_iter${it}/aa_distribution_split0.csv"
            [[ -f "$AA_CSV" ]] || { echo "[warn] missing $AA_CSV"; continue; }
            DEST="${RESULT_BASE}/${ASSAY}_${TAG}_iter${it}"; mkdir -p "$DEST"
            # scorer forces output under  <cwd>/results/<output_dir>/
            WORK="${DEST}/_scorework"; mkdir -p "$WORK"
            ( cd "$WORK" && python "${SRCDIR}/scripts/result_eval_uniform_uaa.py" \
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
            python "${SRCDIR}/scripts/aggregate_splits.py" \
                --glob "${SAMPLE_BASE}/run${TAG}/${P}_${TAG}_${NUM_SAMPLES}_iter*/aa_distribution_split0.csv" \
                --output-dir "$RD" --baselines "$BCSV" --total-num "$NUM_SAMPLES" || true
        fi
    done

    # --- build modeldata.csv (+ collate baselines_pg) ---
    python "${SRCDIR}/eval/make_modeldata.py" \
        --result-base "$RESULT_BASE" --model-tag "$TAG" \
        --num-samples "$NUM_SAMPLES" --out-dir "$MODELDATA_DIR"
done

# --- stage 5: plot the assigned group ---
echo "=== plotting group: $MODELS ==="
python "${SRCDIR}/scripts/compare_models.py" --data-dir "$MODELDATA_DIR" --output-dir "$FIG_DIR"
echo "[done] figures -> $FIG_DIR"
ls -1 "$FIG_DIR"/*.svg
