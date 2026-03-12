#!/bin/bash
set -euo pipefail

# Submit 60 shard jobs, then submit merge job with afterok dependency.
# Usage:
#   ./submit_eqgat_lmdb_60jobs.sh [INPUT_PDB_DIR] [OUTPUT_LMDB_NAME]
#
# Examples:
#   ./submit_eqgat_lmdb_60jobs.sh /scratch/project_465002574/PDB/PDB_cleaned uaag2_eqgat_duallat_all
#   ./submit_eqgat_lmdb_60jobs.sh /scratch/project_465002574/PDB/PDB_cleaned uaag2_eqgat_duallat_all.lmdb
#
# If args are omitted, values fall back to env vars/defaults used by sbatch scripts.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	sed -n '1,20p' "$0"
	exit 0
fi

INPUT_PDB_DIR="${1:-${PDB_DIR:-/scratch/project_465002574/PDB/PDB_cleaned}}"
OUTPUT_LMDB_NAME_RAW="${2:-${FINAL_PREFIX:-uaag2_eqgat_duallat_all}}"

# Normalize optional ".lmdb" suffix because merge script appends it.
OUTPUT_LMDB_NAME="${OUTPUT_LMDB_NAME_RAW%.lmdb}"
if [[ -z "$OUTPUT_LMDB_NAME" ]]; then
	echo "Error: OUTPUT_LMDB_NAME cannot be empty." >&2
	exit 1
fi

# Keep shard naming aligned with selected final LMDB name by default.
OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-${OUTPUT_LMDB_NAME}_shard}"

echo "Using PDB_DIR: ${INPUT_PDB_DIR}"
echo "Using FINAL_PREFIX: ${OUTPUT_LMDB_NAME}"
echo "Using OUTPUT_PREFIX_BASE: ${OUTPUT_PREFIX_BASE}"

ARRAY_JOB_ID=$(sbatch \
	--export=ALL,PDB_DIR="${INPUT_PDB_DIR}",OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE}",FINAL_PREFIX="${OUTPUT_LMDB_NAME}" \
	run_build_eqgat_lmdb_array_sbatch.sh | awk '{print $4}')
echo "Submitted shard array job: ${ARRAY_JOB_ID}"

MERGE_JOB_ID=$(sbatch \
	--dependency=afterok:${ARRAY_JOB_ID} \
	--export=ALL,PDB_DIR="${INPUT_PDB_DIR}",OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE}",FINAL_PREFIX="${OUTPUT_LMDB_NAME}" \
	run_merge_eqgat_lmdb_sbatch.sh | awk '{print $4}')
echo "Submitted merge job: ${MERGE_JOB_ID} (afterok:${ARRAY_JOB_ID})"
