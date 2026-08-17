#!/bin/bash
# One forced-condition CP2 sampling run, pinned to a single GCD.
# TASK_ID indexes the 40 conditions (20 natural + 20 CP2 NCAAs, DAL excluded).
set -u

CODES=(ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
       2NP 2TH 3TH ABU AHP AIB AOC BZT CHA CPA HSM MEA MEB MEF MEG NLE NVA TBU TME YME)
N=${#CODES[@]}

if [[ "$TASK_ID" -ge "$N" ]]; then
    echo "[rank $SLURM_PROCID] task $TASK_ID >= $N — no work"
    exit 0
fi
CODE=${CODES[$TASK_ID]}

export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}_${SLURM_PROCID}"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"

echo "[$(date)] rank ${SLURM_PROCID} gcd ${SLURM_LOCALID}: condition ${CODE}"

python ${SRCDIR}/scripts/generate_ligand_dpm.py \
    --load-ckpt   "$CKPT" \
    --id          "${TAG:-cp2_cond}/${CODE}" \
    --benchmark-path "$BENCH_PT" \
    --save-dir    "$SAMPLE_BASE" \
    --num-samples 100 \
    --split_index 0 --total_partition 1 \
    --every-k-step 25 \
    --ctmc --dpm-solver-pp \
    --batch-size 8 \
    --force-condition "$CODE" \
    --force-condition-table "$COND_TABLE" \
    ${EXTRA_ARGS:-}
rc=$?
rm -rf "$MIOPEN_USER_DB_PATH"
[[ $rc -ne 0 ]] && { echo "[ERROR] condition ${CODE} failed: $rc" >&2; exit $rc; }
echo "[$(date)] rank ${SLURM_PROCID}: condition ${CODE} done"
