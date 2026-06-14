#!/bin/bash
# ============================================================================
# run_eval.sh — ONE-BUTTON UNAAGI evaluation launcher.
# ============================================================================
# Usage (on LUMI, from the UAAG2 repo root):
#   eval/run_eval.sh <model_name> [<model_name> ...]
#   eval/run_eval.sh all
#
# For each model it:
#   1. looks the model up in eval/models.tsv
#   2. resolves SRCDIR for the model's architecture (provisioning a pre-ring
#      git worktree on demand) — see models.tsv for why this matters
#   3. sbatch's the 135-task CTMC 1k x 5iter sampling array (eval/slurm_sample.sh)
#
# After sampling finishes, run the downstream stages (see eval/RUNBOOK.md):
#   PoseBusters QC -> post_analysis -> result_eval_uniform.py / _uaa.py
# ============================================================================
set -euo pipefail

REPO=/flash/project_465002574/UAAG2_main          # ring_membership checkout (live)
PRERING_WT=/flash/project_465002574/UAAG2_preRing  # pre-ring worktree (auto-created)
MANIFEST="$(cd "$(dirname "$0")" && pwd)/models.tsv"
SAMPLE_SH="$(cd "$(dirname "$0")" && pwd)/slurm_sample.sh"

[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST"; exit 1; }

resolve_srcdir() {  # $1=arch $2=code_ref  -> echoes SRCDIR
    local arch="$1" ref="$2"
    if [[ "$arch" == "ring" ]]; then echo "$REPO"; return; fi
    # prering: ensure a worktree pinned at the given commit
    if [[ ! -d "$PRERING_WT" ]]; then
        git -C "$REPO" worktree add "$PRERING_WT" "$ref" >&2
    fi
    echo "$PRERING_WT"
}

launch_one() {  # $1 = model name
    local want="$1" name ckpt arch ref notes
    while IFS=$'\t' read -r name ckpt arch ref notes; do
        [[ "$name" =~ ^#|^name$|^$ ]] && continue
        [[ "$name" == "$want" ]] || continue
        [[ -f "$ckpt" ]] || { echo "[$name] CKPT missing: $ckpt"; return 1; }
        local srcdir; srcdir=$(resolve_srcdir "$arch" "$ref")
        echo "[$name] arch=$arch  srcdir=$srcdir"
        echo "[$name] ckpt=$ckpt"
        local jid
        jid=$(sbatch --parsable --job-name="eval_${name}" \
              --export=ALL,CKPT="$ckpt",MODEL_TAG="$name",SRCDIR="$srcdir" \
              "$SAMPLE_SH")
        echo "[$name] submitted sampling array: job $jid (135 tasks, throttle %30)"
        echo "$name $jid" >> /scratch/project_465002574/uaag2_adit_results/eval_jobids.txt
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
