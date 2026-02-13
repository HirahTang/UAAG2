#!/bin/bash
#SBATCH --job-name=pdb_clean
#SBATCH --account=project_465002574
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/pdb_clean_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/pdb_clean_%j.log

set -euo pipefail

echo "============================================================================"
echo "PDB CIF Cleaning"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================================"

# Optional: load modules / activate env if needed
# module load LUMI
# module load CrayEnv
# module load lumi-container-wrapper/0.4.2-cray-python-default
# export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

PDB_DIR="/scratch/project_465002574/PDB/PDB"
CLEANED_DIR="/scratch/project_465002574/PDB/PDB_cleaned"

mkdir -p "${CLEANED_DIR}"

echo "Input CIF dir:  ${PDB_DIR}"
echo "Output PDB dir: ${CLEANED_DIR}"

python scripts/pdb_clean.py \
    --pdb_dir "${PDB_DIR}" \
    --cleaned_dir "${CLEANED_DIR}"

echo "============================================================================"
echo "PDB cleaning completed: $(date)"
echo "============================================================================"
