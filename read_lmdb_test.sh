#!/bin/bash
#SBATCH --job-name=checkup
#SBATCH --account=project_465002574
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/check.log
#SBATCH -e logs/check.log
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

export WORK_DIR=/flash/project_465002574/UAAG2_main

python read_lmdb_test.py