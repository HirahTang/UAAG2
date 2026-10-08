#!/bin/bash
# One v6 conditional sampling run on Hendrix, single GPU.
# Env in: ARM (grover|morgan), STAGE (cond|cfg|nopocket), GLOBAL (task index)
set -u

CODES=(ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
       2NP 2TH 3TH ABU AHP AIB AOC BZT CHA CPA HSM MEA MEB MEF MEG NLE NVA TBU TME YME)
# heavy-atom count per code, derived from the eval table's own SMILES (same order)
NODES=(5 11 8 8 6 9 9 4 10 8 8 9 8 11 7 6 7 14 12 7
       15 10 10 6 10 6 10 14 11 10 8 6 7 12 5 8 7 9 8 13)
N=${#CODES[@]}

SCALES=(1.0 1.5 2.0 3.0 5.0)

if [ "$STAGE" = "cfg" ]; then
    IDX=$(( GLOBAL % N ))
    SI=$(( GLOBAL / N ))
    W=${SCALES[$SI]}
    TAG="v6_${ARM}_cfg_w${W}"
    EXTRA="--cfg-scale ${W}"
else
    IDX=$GLOBAL
    W="-"
    TAG="v6_${ARM}_${STAGE}"
    EXTRA=""
fi

if [ "$IDX" -ge "$N" ]; then echo "[skip] idx $IDX >= $N"; exit 0; fi
CODE=${CODES[$IDX]}
NODE=${NODES[$IDX]}

R=/home/qcx679/hantang/UAAG2_conditional
CE=/datasets/biochem/unaagi/condition_embeddings
BD=/home/qcx679/hantang/UAAG2/data/full_graph/benchmarks

CKPT=/datasets/biochem/unaagi/ckpts_v6/v6_${ARM}_ep199.ckpt
if [ "$ARM" = "grover" ]; then
    COND_TABLE=$CE/cp2_experiment_conditions.npz
    TRAIN_TABLE=$CE/grover_base_condition_table.npz
else
    COND_TABLE=$CE/cp2_experiment_conditions_morgan.npz
    TRAIN_TABLE=$CE/morgan_r3_2048_condition_table.npz
fi

if [ "$STAGE" = "nopocket" ]; then
    BENCH=$BD/5ly1_cp2_nopocket.pt
else
    BENCH=$BD/5ly1_cp2.pt
fi

SAMPLE_BASE=/datasets/biochem/unaagi/samples_v6/${ARM}_${STAGE}
mkdir -p "$SAMPLE_BASE"

echo "[$(date)] ARM=$ARM STAGE=$STAGE GLOBAL=$GLOBAL code=$CODE nodes=$NODE w=$W"

cd $R
python scripts/generate_ligand_dpm.py \
    --load-ckpt   "$CKPT" \
    --id          "${TAG}/${CODE}" \
    --benchmark-path "$BENCH" \
    --data_info_path /home/qcx679/hantang/UAAG2/data/full_graph/statistic.pkl \
    --save-dir    "$SAMPLE_BASE" \
    --num-samples 100 \
    --split_index 0 --total_partition 1 \
    --every-k-step 25 \
    --ctmc --dpm-solver-pp \
    --batch-size 8 \
    --virtual_node_size "$NODE" --exact-node-count \
    --cond-table "$TRAIN_TABLE" \
    --force-condition "$CODE" \
    --force-condition-table "$COND_TABLE" \
    $EXTRA
rc=$?
[ $rc -ne 0 ] && { echo "[ERROR] $ARM/$STAGE/$CODE rc=$rc" >&2; exit $rc; }
echo "[$(date)] done $ARM/$STAGE/$CODE"
