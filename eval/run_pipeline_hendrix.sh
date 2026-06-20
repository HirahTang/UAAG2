#!/bin/bash
# ============================================================================
# run_pipeline_hendrix.sh — THE ONE BUTTON on HENDRIX (CUDA).
# Full UNAAGI eval for a group of models:
#     sampling -> post-analysis -> aggregate + Spearman -> visualize group
# chained with SLURM afterok dependencies.
#
# Usage (Hendrix, from a checkout that has this branch, e.g. /home/qcx679/hantang/UAAG2):
#   eval/run_pipeline_hendrix.sh v0.3ring_base v0.3ring_cont v0.3weighted_end
#   eval/run_pipeline_hendrix.sh all
#
#   1. per model: sbatch the 135-task sampling array (eval/slurm_sample_hendrix.sh),
#      with the right CKPT/SRCDIR from eval/models_hendrix.tsv. Sampling also does
#      post-analysis (aa_distribution) + inline ProteinGym aggregate.
#   2. sbatch eval/finalize_and_plot_hendrix.sh with afterok on ALL sampling arrays:
#      CP2/PUMA per-iter scoring -> <TAG>_modeldata.csv per model -> compare_models.py.
#
# Figures: /datasets/biochem/unaagi/results/figures/pipeline_<timestamp>/
# Full procedure: eval/PIPELINE.md.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"      # resolves on the login node (run directly)
REPO="$(cd "$HERE/.." && pwd)"             # repo root holding eval/ + scripts/
MANIFEST="$HERE/models_hendrix.tsv"
SAMPLE_SH="$HERE/slurm_sample_hendrix.sh"
FINAL_SH="$HERE/finalize_and_plot_hendrix.sh"
JOBLOG=/datasets/biochem/unaagi/eval_jobids.txt

[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST"; exit 1; }
[[ $# -ge 1 ]] || { echo "usage: $0 <model> [<model> ...] | all"; exit 1; }

if [[ "$1" == "all" ]]; then
    mapfile -t MODELS < <(awk -F'\t' '!/^#/ && $1!="name" && NF>=4 {print $1}' "$MANIFEST")
else
    MODELS=("$@")
fi

lookup() {  # $1=name -> "ckpt\tarch\tsrcdir"
    awk -F'\t' -v w="$1" '!/^#/ && $1==w {print $2"\t"$3"\t"$4; exit}' "$MANIFEST"
}

SAMPLE_JIDS=()
for m in "${MODELS[@]}"; do
    row=$(lookup "$m"); [[ -n "$row" ]] || { echo "[skip] $m not in manifest"; continue; }
    IFS=$'\t' read -r ckpt arch srcdir <<<"$row"
    [[ -f "$ckpt" ]]   || { echo "[skip] $m ckpt missing: $ckpt"; continue; }
    [[ -d "$srcdir" ]] || { echo "[skip] $m srcdir missing: $srcdir"; continue; }
    jid=$(sbatch --parsable --job-name="samp_${m}" \
          --export=ALL,CKPT="$ckpt",MODEL_TAG="$m",SRCDIR="$srcdir" "$SAMPLE_SH")
    echo "[$m] sampling array job $jid (arch=$arch)"
    echo "$m $jid $(date)" >> "$JOBLOG"
    SAMPLE_JIDS+=("$jid")
done

[[ ${#SAMPLE_JIDS[@]} -ge 1 ]] || { echo "no models submitted"; exit 1; }

DEP=$(IFS=:; echo "afterok:${SAMPLE_JIDS[*]}")
FJID=$(sbatch --parsable --job-name="finalize" --dependency="$DEP" \
       --export=ALL,MODELS="${MODELS[*]}",PIPE_SRC="$REPO" "$FINAL_SH")
echo
echo "[finalize+plot] job $FJID  (runs after: $DEP)"
echo "group: ${MODELS[*]}"
echo "figures -> /datasets/biochem/unaagi/results/figures/pipeline_<timestamp>/"
