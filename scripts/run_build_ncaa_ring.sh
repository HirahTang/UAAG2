#!/bin/bash
#SBATCH --job-name=build_ncaa_ring
#SBATCH --account=project_465002574
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/project_465002574/logs/build_ncaa_ring_%j.log

module load LUMI
module load CrayEnv
export PATH=/flash/project_465002574/unaagi_env/bin:$PATH
export PYTHONPATH=/flash/project_465002574/UAAG2_main/src:$PYTHONPATH

python /flash/project_465002574/UAAG2_main/scripts/build_ncaa_lmdb.py \
    --in-dir  /scratch/project_465002574/PDB/naa \
    --out-dir /scratch/project_465002574/PDB/NCAA_new_with_ring
