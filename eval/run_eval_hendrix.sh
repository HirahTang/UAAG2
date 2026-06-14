#!/bin/bash
# ============================================================================
# run_eval_hendrix.sh — ONE-BUTTON UNAAGI evaluation launcher for HENDRIX (CUDA).
# ============================================================================
# Usage (on Hendrix, from /home/qcx679/hantang/UAAG2):
#   eval/run_eval_hendrix.sh <model_name> [<model_name> ...]
#   eval/run_eval_hendrix.sh all
#
# Reads eval/models_hendrix.tsv, and for each model sbatch's the 135-task
# CTMC 1k x 5iter sampling array (eval/slurm_sample_hendrix.sh) with the right
# CKPT and SRCDIR (ring worktree vs pre-ring main). See eval/RUNBOOK.md.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$HERE/models_hendrix.tsv"
SAMPLE_SH="$HERE/slurm_sample_hendrix.sh"
JOBIDS=/datasets/biochem/unaagi/eval_jobids.txt
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST"; exit 1; }

launch_one() {
    local want="$1" name ckpt arch srcdir notes
    while IFS=$'\t' read -r name ckpt arch srcdir notes; do
        [[ "$name" =~ ^#|^name$|^$ ]] && continue
        [[ "$name" == "$want" ]] || continue
        [[ -f "$ckpt" ]]   || { echo "[$name] CKPT missing: $ckpt"; return 1; }
        [[ -d "$srcdir" ]] || { echo "[$name] SRCDIR missing: $srcdir"; return 1; }
        echo "[$name] arch=$arch srcdir=$srcdir"
        local jid
        jid=$(sbatch --parsable --job-name="$name" \
              --export=ALL,CKPT="$ckpt",MODEL_TAG="$name",SRCDIR="$srcdir" \
              "$SAMPLE_SH")
        echo "[$name] submitted sampling array: job $jid"
        echo "$name $jid $(date)" >> "$JOBIDS"
        return 0
    done < "$MANIFEST"
    echo "model not found in manifest: $want"; return 1
}

[[ $# -ge 1 ]] || { echo "usage: $0 <model_name> [...] | all"; exit 1; }
if [[ "$1" == "all" ]]; then
    mapfile -t names < <(awk -F'\t' '!/^#/ && $1!="name" && NF>=4 {print $1}' "$MANIFEST")
    set -- "${names[@]}"
fi
for m in "$@"; do launch_one "$m"; done
