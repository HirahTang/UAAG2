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

# Runtime tuning knobs (override via env when needed):
#   NUM_SHARDS=240 MAX_CONCURRENT_SHARDS=30 SHARD_MEM=64G SHARD_CPUS=4 ./submit_eqgat_lmdb_60jobs.sh ...
NUM_SHARDS="${NUM_SHARDS:-120}"
MAX_CONCURRENT_SHARDS="${MAX_CONCURRENT_SHARDS:-40}"
SHARD_MEM="${SHARD_MEM:-64G}"
SHARD_CPUS="${SHARD_CPUS:-4}"
LATENT_CACHE_FILES="${LATENT_CACHE_FILES:-16}"
WRITE_METADATA="${WRITE_METADATA:-0}"
LOG_MEMORY_EVERY="${LOG_MEMORY_EVERY:-25}"
GC_EVERY="${GC_EVERY:-50}"
MERGE_MEM="${MERGE_MEM:-96G}"
MERGE_SKIP_METADATA="${MERGE_SKIP_METADATA:-1}"

if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARDS" -lt 1 ]]; then
	echo "Error: NUM_SHARDS must be a positive integer." >&2
	exit 1
fi
if ! [[ "$MAX_CONCURRENT_SHARDS" =~ ^[0-9]+$ ]] || [[ "$MAX_CONCURRENT_SHARDS" -lt 1 ]]; then
	echo "Error: MAX_CONCURRENT_SHARDS must be a positive integer." >&2
	exit 1
fi
if ! [[ "$LATENT_CACHE_FILES" =~ ^[0-9]+$ ]] || [[ "$LATENT_CACHE_FILES" -lt 1 ]]; then
	echo "Error: LATENT_CACHE_FILES must be a positive integer." >&2
	exit 1
fi
if ! [[ "$LOG_MEMORY_EVERY" =~ ^[0-9]+$ ]]; then
	echo "Error: LOG_MEMORY_EVERY must be a non-negative integer." >&2
	exit 1
fi
if ! [[ "$GC_EVERY" =~ ^[0-9]+$ ]]; then
	echo "Error: GC_EVERY must be a non-negative integer." >&2
	exit 1
fi

ARRAY_SPEC="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT_SHARDS}"

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
echo "Using NUM_SHARDS: ${NUM_SHARDS}"
echo "Using array spec: ${ARRAY_SPEC}"
echo "Using SHARD_MEM: ${SHARD_MEM}, SHARD_CPUS: ${SHARD_CPUS}"
echo "Using LATENT_CACHE_FILES: ${LATENT_CACHE_FILES}"
echo "Using WRITE_METADATA: ${WRITE_METADATA}"
echo "Using LOG_MEMORY_EVERY: ${LOG_MEMORY_EVERY}"
echo "Using GC_EVERY: ${GC_EVERY}"
echo "Using MERGE_MEM: ${MERGE_MEM}"
echo "Using MERGE_SKIP_METADATA: ${MERGE_SKIP_METADATA}"

ARRAY_JOB_ID=$(sbatch \
	--array="${ARRAY_SPEC}" \
	--mem="${SHARD_MEM}" \
	--cpus-per-task="${SHARD_CPUS}" \
	--export=ALL,PDB_DIR="${INPUT_PDB_DIR}",OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE}",FINAL_PREFIX="${OUTPUT_LMDB_NAME}",NUM_SHARDS="${NUM_SHARDS}",LATENT_CACHE_FILES="${LATENT_CACHE_FILES}",WRITE_METADATA="${WRITE_METADATA}",LOG_MEMORY_EVERY="${LOG_MEMORY_EVERY}",GC_EVERY="${GC_EVERY}" \
	run_build_eqgat_lmdb_array_sbatch.sh | awk '{print $4}')
echo "Submitted shard array job: ${ARRAY_JOB_ID}"

MERGE_JOB_ID=$(sbatch \
	--dependency=afterok:${ARRAY_JOB_ID} \
	--mem="${MERGE_MEM}" \
	--export=ALL,PDB_DIR="${INPUT_PDB_DIR}",OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE}",FINAL_PREFIX="${OUTPUT_LMDB_NAME}",SKIP_METADATA="${MERGE_SKIP_METADATA}" \
	run_merge_eqgat_lmdb_sbatch.sh | awk '{print $4}')
echo "Submitted merge job: ${MERGE_JOB_ID} (afterok:${ARRAY_JOB_ID})"
