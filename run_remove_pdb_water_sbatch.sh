#!/bin/bash
#SBATCH --job-name=remove_pdb_water
#SBATCH --account=project_465002574
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/remove_pdb_water_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/remove_pdb_water_%j.log

set -euo pipefail

echo "============================================================================"
echo "PDB Water Removal"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "============================================================================"

# Optional module setup for LUMI; safe to keep if modules are available.
module load LUMI || true
module load CrayEnv || true

REPO_DIR=${REPO_DIR:-/flash/project_465002574/UAAG2_main}
SCRIPT_PATH=${SCRIPT_PATH:-${REPO_DIR}/extract_pdb_water.sh}

INPUT_DIR=${INPUT_DIR:-/scratch/project_465002574/PDB/PDB_cleaned}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/project_465002574/PDB/PDB_water}

mkdir -p /scratch/project_465002574/UAAG_logs

if [[ ! -x "$SCRIPT_PATH" ]]; then
  echo "Script not executable or missing: $SCRIPT_PATH"
  echo "Trying to set executable bit..."
  chmod +x "$SCRIPT_PATH"
fi

echo "Repo dir:   $REPO_DIR"
echo "Script:     $SCRIPT_PATH"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"

cd "$REPO_DIR"

"$SCRIPT_PATH" "$INPUT_DIR" "$OUTPUT_DIR"

echo "============================================================================"
echo "Water removal completed: $(date)"
echo "============================================================================"
