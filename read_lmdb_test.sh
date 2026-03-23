#!/bin/bash
#SBATCH --job-name=checkup
#SBATCH --account=project_465002574
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/check_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/check_%j.log

module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

WORK_DIR=/flash/project_465002574/UAAG2_main
DATA_PATH=/scratch/project_465002574/unaagi_whole_v1.lmdb
OUT_DIR=/scratch/project_465002574/UAAG_logs/lmdb_stats

mkdir -p ${OUT_DIR}
cd ${WORK_DIR}

python read_lmdb_test.py \
	--data-path ${DATA_PATH} \
	--csv-out ${OUT_DIR}/lmdb_aa_frequency.csv \
	--summary-out ${OUT_DIR}/lmdb_aa_summary.csv \
	--hist-out ${OUT_DIR}/lmdb_aa_frequency_hist.png
