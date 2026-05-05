#!/bin/bash
#SBATCH --job-name=build_pdbbind
#SBATCH --account=project_465002574
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/scratch/project_465002574/logs/build_pdbbind_%j.log

module load LUMI
module load CrayEnv
export PATH=/flash/project_465002574/unaagi_env/bin:$PATH
export PYTHONPATH=/flash/project_465002574/UAAG2_main/src:$PYTHONPATH

python /flash/project_465002574/UAAG2_main/scripts/build_pdbbind_lmdb.py     --in-dir  /scratch/project_465002574/PDB/PDBBind/pbpp-2020     --out-path /scratch/project_465002574/PDB/PDBBind.lmdb     --n-workers 14     --edge-radius 8.0     --log-path /scratch/project_465002574/logs/build_pdbbind_errors.log
