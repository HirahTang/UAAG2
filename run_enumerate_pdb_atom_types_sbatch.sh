#!/bin/bash
#SBATCH --job-name=PDB_atom_enum
#SBATCH --account=project_465002574
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/PDB_atom_enum_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/PDB_atom_enum_%j.log

set -euo pipefail

echo "============================================================================"
echo "PDB Atom Type Enumeration"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================================"

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default

# Use the same python environment style as existing jobs
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

# Go to repo root
REPO_DIR="/flash/project_465002574/UAAG2_main"
cd "$REPO_DIR"

echo "Running on branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Commit hash:       $(git rev-parse HEAD 2>/dev/null || echo n/a)"

ROOT_DIR="/scratch/project_465002574/PDB/PDB_cleaned"
OUT_DIR="/scratch/project_465002574/PDB/atom_type_summary"

mkdir -p "$OUT_DIR"

OUT_JSON="$OUT_DIR/pdb_atom_report_${SLURM_JOB_ID}.json"
OUT_CSV="$OUT_DIR/atom_names_${SLURM_JOB_ID}.csv"

echo "Input root:  $ROOT_DIR"
echo "Output JSON: $OUT_JSON"
echo "Output CSV:  $OUT_CSV"

python scripts/enumerate_pdb_atom_types.py \
    --root "$ROOT_DIR" \
    --out-json "$OUT_JSON" \
    --out-csv "$OUT_CSV" \
    --top 0

echo ""
echo "============================================================================"
echo "Enumeration Completed Successfully"
echo "============================================================================"
echo "End time: $(date)"
echo "Results saved to: $OUT_DIR"
echo "============================================================================"
