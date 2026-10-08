#!/bin/bash
#SBATCH --job-name=v6eval
#SBATCH --partition=gpu,boomsma
#SBATCH --gres=gpu:1
#SBATCH --exclude=hendrixgpu01fl,hendrixgpu04fl,hendrixgpu06fl,hendrixgpu16fl,hendrixgpu19fl,hendrixgpu24fl,hendrixgpu25fl,hendrixgpu26fl
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/datasets/biochem/unaagi/samples_v6/logs/v6_%x_%A_%a.out
#SBATCH --error=/datasets/biochem/unaagi/samples_v6/logs/v6_%x_%A_%a.out
#
# v6 conditional evaluation, Hendrix port of the LUMI four-stage pipeline
# (uncond dropped -- v6 is pocket-conditional, the scoring axis no longer applies).
#
# Submit with:  sbatch --job-name=<arm>_<stage> --array=<range> \
#                      --export=ALL,ARM=<grover|morgan>,STAGE=<cond|cfg|nopocket> submit_v6.sbatch
#   cond     --array=0-39    (40 conditions, w=1.0)
#   cfg      --array=0-199   (5 scales x 40)
#   nopocket --array=0-39    (40 conditions, pocket-free benchmark)
#
# The all-atom sampling patch (uaag_dataset.py ~457) is driven by the ckpt hparam
# all_atom_generation=True, so no flag here -- v5.x ckpts are unaffected.
set -uo pipefail
source /opt/software/mamba/1.4.1/etc/profile.d/conda.sh
conda activate targetdiff
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/qcx679/hantang/UAAG2_conditional/src:${PYTHONPATH:-}

mkdir -p /datasets/biochem/unaagi/samples_v6/logs

export GLOBAL=${SLURM_ARRAY_TASK_ID}
bash /home/qcx679/hantang/UAAG2_conditional/slurm_hx/v6_worker.sh
