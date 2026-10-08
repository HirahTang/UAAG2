#!/bin/bash
#SBATCH --job-name=pbjoin
#SBATCH --partition=gpu,boomsma
#SBATCH --exclude=hendrixgpu01fl,hendrixgpu04fl,hendrixgpu06fl,hendrixgpu16fl,hendrixgpu19fl,hendrixgpu24fl,hendrixgpu25fl,hendrixgpu26fl
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/datasets/biochem/unaagi/samples_v6/pbjoin/logs/pb_%A_%a.out
set -uo pipefail
source /opt/software/mamba/1.4.1/etc/profile.d/conda.sh
conda activate targetdiff
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/qcx679/hantang/UAAG2_conditional/src:${PYTHONPATH:-}

CODES=(ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
       2NP 2TH 3TH ABU AHP AIB AOC BZT CHA CPA HSM MEA MEB MEF MEG NLE NVA TBU TME YME)
STAGES=("grover_cond/runv6_grover_cond"
        "grover_cfg/runv6_grover_cfg_w2.0"
        "grover_nopocket/runv6_grover_nopocket"
        "morgan_cond/runv6_morgan_cond"
        "morgan_cfg/runv6_morgan_cfg_w2.0"
        "morgan_nopocket/runv6_morgan_nopocket")

N=${#CODES[@]}                       # 40
G=$SLURM_ARRAY_TASK_ID
SI=$(( G / N )); CI=$(( G % N ))
export STAGE_SUB="${STAGES[$SI]}"
export CODE="${CODES[$CI]}"
TAG=$(echo "$STAGE_SUB" | tr '/' '_')
export OUT="/datasets/biochem/unaagi/samples_v6/pbjoin/${TAG}__${CODE}.json"

if [ -s "$OUT" ]; then
  echo "[skip] $OUT already present"; exit 0
fi
echo "[$(date)] stage=$STAGE_SUB code=$CODE -> $OUT"
python /home/qcx679/hantang/v6_analysis/pb_join.py
