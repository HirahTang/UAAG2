#!/bin/bash
#SBATCH --job-name=eqgat_lmdb_merge
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o /scratch/project_465002574/UAAG_logs/eqgat_lmdb_merge_%j.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/eqgat_lmdb_merge_%j.log

set -euo pipefail

echo "============================================================================"
echo "UAAG2 EQGAT LMDB Merge"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "============================================================================"

module load LUMI || true
module load CrayEnv || true

PYTHON_BIN=${PYTHON_BIN:-/flash/project_465002574/unaagi_env/bin/python}
REPO_DIR=${REPO_DIR:-/flash/project_465002574/UAAG2_main}

OUTPUT_DIR=${OUTPUT_DIR:-/scratch/project_465002574/PDB/uaag2_eqgat_lmdb_shards}
OUTPUT_PREFIX_BASE=${OUTPUT_PREFIX_BASE:-uaag2_eqgat_duallat_shard}
FINAL_PREFIX=${FINAL_PREFIX:-uaag2_eqgat_duallat_all}

SHARD_GLOB="${OUTPUT_DIR}/${OUTPUT_PREFIX_BASE}_*.lmdb"
FINAL_LMDB="${OUTPUT_DIR}/${FINAL_PREFIX}.lmdb"
FINAL_META="${OUTPUT_DIR}/${FINAL_PREFIX}.metadata.pkl"
SKIP_METADATA="${SKIP_METADATA:-1}"

MERGE_ARGS=(
  --shard_glob "$SHARD_GLOB"
  --output_lmdb "$FINAL_LMDB"
  --output_metadata "$FINAL_META"
)
if [[ "$SKIP_METADATA" == "1" ]]; then
  MERGE_ARGS+=(--skip_metadata)
fi

cd "$REPO_DIR"

echo "Python: $PYTHON_BIN"
echo "Merging shards: $SHARD_GLOB"
echo "Final LMDB: $FINAL_LMDB"

time "$PYTHON_BIN" scripts/merge_lmdb_shards.py "${MERGE_ARGS[@]}"

echo "End time: $(date)"
echo "Merge completed"
