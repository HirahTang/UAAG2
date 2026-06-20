#!/bin/bash
# ============================================================================
# run_pipeline.sh — THE ONE BUTTON.
# Full UNAAGI eval for a group of models:
#     sampling -> post-analysis -> aggregate + Spearman -> visualize group
# in a single command, chained with SLURM afterok dependencies.
#
# Usage (LUMI, from repo root /flash/project_465002574/UAAG2_main):
#   eval/run_pipeline.sh v0.3ring_base v0.3ring_cont v0.3weighted_end p0203_ctmc
#   eval/run_pipeline.sh all
#
# What it does:
#   1. For each model: sbatch the 135-task sampling array (eval/slurm_sample.sh),
#      resolving the right code checkout per the model's arch (eval/models.tsv).
#      Sampling itself does post-analysis (aa_distribution) + inline ProteinGym aggregate.
#   2. sbatch eval/finalize_and_plot.sh with --dependency=afterok on ALL sampling
#      arrays. It does CP2/PUMA per-iter scoring, builds <TAG>_modeldata.csv for each
#      model, then runs scripts/compare_models.py over the whole group -> SVGs.
#
# Output figures: /scratch/.../UNAAGI_result/figures/pipeline_<timestamp>/
# See eval/PIPELINE.md for the full procedure (an agent can follow that file).
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/flash/project_465002574/UAAG2_main
PRERING_WT=/flash/project_465002574/UAAG2_preRing
MANIFEST="$HERE/models.tsv"
SAMPLE_SH="$HERE/slurm_sample.sh"
FINAL_SH="$HERE/finalize_and_plot.sh"
JOBLOG=/scratch/project_465002574/uaag2_adit_results/eval_jobids.txt

[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST"; exit 1; }
[[ $# -ge 1 ]] || { echo "usage: $0 <model> [<model> ...] | all"; exit 1; }

if [[ "$1" == "all" ]]; then
    mapfile -t MODELS < <(awk -F'\t' '!/^#/ && $1!="name" && NF>=4 {print $1}' "$MANIFEST")
else
    MODELS=("$@")
fi

resolve_srcdir() {  # $1=arch $2=ref
    local arch="$1" ref="$2"
    if [[ "$arch" == "ring" ]]; then echo "$REPO"; return; fi
    [[ -d "$PRERING_WT" ]] || git -C "$REPO" worktree add "$PRERING_WT" "$ref" >&2
    echo "$PRERING_WT"
}
lookup() {  # $1=name -> sets row vars via global; echoes "ckpt\tarch\tref"
    awk -F'\t' -v w="$1" '!/^#/ && $1==w {print $2"\t"$3"\t"$4; exit}' "$MANIFEST"
}

SAMPLE_JIDS=()
RING_SRC=""        # SRCDIR for the finalize job (ring checkout has compare_models + make_modeldata)
for m in "${MODELS[@]}"; do
    row=$(lookup "$m"); [[ -n "$row" ]] || { echo "[skip] $m not in manifest"; continue; }
    IFS=$'\t' read -r ckpt arch ref <<<"$row"
    [[ -f "$ckpt" ]] || { echo "[skip] $m ckpt missing: $ckpt"; continue; }
    srcdir=$(resolve_srcdir "$arch" "$ref"); RING_SRC=${RING_SRC:-$REPO}
    jid=$(sbatch --parsable --job-name="samp_${m}" \
          --export=ALL,CKPT="$ckpt",MODEL_TAG="$m",SRCDIR="$srcdir" "$SAMPLE_SH")
    echo "[$m] sampling array job $jid (arch=$arch)"
    echo "$m $jid" >> "$JOBLOG"
    SAMPLE_JIDS+=("$jid")
done

[[ ${#SAMPLE_JIDS[@]} -ge 1 ]] || { echo "no models submitted"; exit 1; }

DEP=$(IFS=:; echo "afterok:${SAMPLE_JIDS[*]}")
FJID=$(sbatch --parsable --job-name="finalize" --dependency="$DEP" \
       --export=ALL,MODELS="${MODELS[*]}",SRCDIR="$RING_SRC" "$FINAL_SH")
echo
echo "[finalize+plot] job $FJID  (runs after: $DEP)"
echo "group: ${MODELS[*]}"
echo "figures will land in /scratch/project_465002574/UNAAGI_result/figures/pipeline_<timestamp>/"
