#!/bin/bash

# ============================================================================
# UAAG Swift Pipeline Submission Script (Single-File)
# ============================================================================
# Submits swift-mode jobs for UAAG assays:
#   Sampling (1 iteration × 10 splits, 100 samples/split)
#   -> Post-analysis + evaluation
#   -> Cleanup (compress + delete .mol/.sdf)
# PoseBusters is skipped in this mode.
# All job logic is embedded in this file (no nested bash dependencies).
#
# Usage:
#   ./submit_assay_job_swift.sh <assay_name|all> <checkpoint_path> [num_samples] [model_name] [split_count]
# ============================================================================

ASSAY_DIR="slurm_config/assays"
WORK_DIR="/flash/project_465002574/UAAG2_main"
LOG_DIR="/scratch/project_465002574/UAAG_logs"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <assay_name|all> <checkpoint_path> [num_samples] [model_name] [split_count]"
    echo ""
    echo "Examples:"
    echo "  $0 ENVZ_ECOLI /flash/project_465002574/UAAG2_main/UAAG_model/last.ckpt"
    echo "  $0 all /flash/project_465002574/UAAG2_main/UAAG_model/last.ckpt"
    echo "  $0 DN7A_SACS2 /path/to/last.ckpt 100 UAAG_model 3"
    exit 1
fi

ASSAY_NAME=$1
CKPT_PATH=$2
NUM_SAMPLES=${3:-100}
MODEL_NAME=${4:-$(basename "$(dirname "${CKPT_PATH}")")}
SPLIT_COUNT=${5:-3}

if ! [[ ${SPLIT_COUNT} =~ ^[0-9]+$ ]]; then
    echo "Error: split_count must be an integer (2 or 3), got: ${SPLIT_COUNT}"
    exit 1
fi

if [ ${SPLIT_COUNT} -lt 2 ] || [ ${SPLIT_COUNT} -gt 3 ]; then
    echo "Error: split_count must be 2 or 3, got: ${SPLIT_COUNT}"
    exit 1
fi

if [ ! -f "${CKPT_PATH}" ]; then
    echo "Error: checkpoint path not found: ${CKPT_PATH}"
    exit 1
fi

submit_assay() {
    local config_file=$1
    local assay_name
    assay_name=$(basename "${config_file}" .txt)
    local protein_id
    local baseline
    local available_splits
    local assay_split_count
    local array_range

    protein_id=$(awk 'NR==2 {print $2}' "${config_file}")
    baseline=$(awk 'NR==2 {print $3}' "${config_file}")
    available_splits=$(awk 'NR>1 && $5==0 {print $4}' "${config_file}" | sort -n | uniq | wc -l | tr -d ' ')

    if [ -z "${available_splits}" ] || [ "${available_splits}" -eq 0 ]; then
        echo "  ✗ No iteration-0 split rows found in ${config_file}"
        return
    fi

    assay_split_count=${SPLIT_COUNT}
    if [ ${available_splits} -lt ${assay_split_count} ]; then
        assay_split_count=${available_splits}
    fi

    array_range="0-$((assay_split_count - 1))"

    if [ -z "${protein_id}" ]; then
        echo "  ✗ Failed to parse protein id from ${config_file}"
        return
    fi

    echo "Submitting SWIFT jobs for: ${assay_name}"
    echo "  Config: ${config_file}"
    echo "  Protein ID: ${protein_id}"
    echo "  Checkpoint: ${CKPT_PATH}"
    echo "  Model: ${MODEL_NAME}"
    echo "  Pipeline: Sampling (1 iter × ${assay_split_count} splits) → Post-analysis/Eval → Cleanup"
    echo "  Samples per split: ${NUM_SAMPLES}"
    echo "  Split count: ${assay_split_count}"
    echo ""

    local sampling_wrap
    sampling_wrap=$(cat <<EOF
set -e
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:\$PATH"
mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

CONFIG_FILE="${config_file}"
CKPT_PATH="${CKPT_PATH}"
MODEL="${MODEL_NAME}"
NUM_SAMPLES="${NUM_SAMPLES}"
SPLIT_COUNT="${assay_split_count}"

ARRAY_ID=\${SLURM_ARRAY_TASK_ID}
PROTEIN_ID=\$(awk 'NR==2 {print \$2}' "\${CONFIG_FILE}")
ITERATION=0
SPLIT_LIST=\$(awk 'NR>1 && \$5==0 {print \$4}' "\${CONFIG_FILE}" | sort -n | uniq | head -n "\${SPLIT_COUNT}" | tr '\n' ' ')
SPLIT_INDEX=\$(echo "\${SPLIT_LIST}" | awk -v i=\${ARRAY_ID} '{print \$(i+1)}')

if [ -z "\${PROTEIN_ID}" ] || [ -z "\${SPLIT_INDEX}" ]; then
    echo "Error: failed to derive protein/split for task \${ARRAY_ID} from \${CONFIG_FILE}"
    exit 1
fi

BENCHMARK_PATH="/scratch/project_465002574/UNAAGI_benchmarks/\${PROTEIN_ID}.pt"
RUN_ID="\${MODEL}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter\${ITERATION}"

python scripts/generate_ligand.py \
    --load-ckpt "\${CKPT_PATH}" \
    --id "\${RUN_ID}" \
    --batch-size 8 \
    --virtual_node_size 15 \
    --num-samples "\${NUM_SAMPLES}" \
    --benchmark-path "\${BENCHMARK_PATH}" \
    --split_index "\${SPLIT_INDEX}" \
    --total_partition "\${SPLIT_COUNT}" \
    --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl
EOF
)

    SAMPLING_JOB=$(sbatch --parsable \
        --account=project_465002574 \
        --partition=standard-g \
        --ntasks=1 \
        --cpus-per-task=7 \
        --gpus-per-node=1 \
        --mem=60G \
        --time=2-00:00:00 \
        --array=${array_range} \
        -o /scratch/project_465002574/UAAG_logs/array_swift_%A_%a.log \
        -e /scratch/project_465002574/UAAG_logs/array_swift_%A_%a.log \
        --job-name="UAAG_swift_samp_${assay_name}" \
        --export=ALL \
        --wrap "${sampling_wrap}" 2>&1)

    if [[ ${SAMPLING_JOB} =~ ^[0-9]+$ ]]; then
        echo "  ✓ Step 1: Sampling submitted: ${SAMPLING_JOB} (${assay_split_count} array tasks: iter0 reduced splits)"

        local postproc_wrap
        postproc_wrap=$(cat <<EOF
set -e
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:\$PATH"
mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

MODEL="${MODEL_NAME}"
NUM_SAMPLES="${NUM_SAMPLES}"
PROTEIN_ID="${protein_id}"
BASELINE="${baseline}"
ITERATION=0

RUN_ID="\${MODEL}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter\${ITERATION}"
SAMPLES_DIR="/scratch/project_465002574/ProteinGymSampling/run\${RUN_ID}/Samples"

if [ ! -d "\${SAMPLES_DIR}" ]; then
    echo "Error: samples directory not found: \${SAMPLES_DIR}"
    exit 1
fi

python scripts/post_analysis.py --analysis_path "\${SAMPLES_DIR}"

AA_DIST="/scratch/project_465002574/ProteinGymSampling/run\${RUN_ID}/aa_distribution.csv"
if [ ! -f "\${AA_DIST}" ]; then
    AA_DIST=\$(find /scratch/project_465002574/ProteinGymSampling/run\${MODEL}/\${PROTEIN_ID}_* -name "aa_distribution.csv" | head -1)
fi

if [ -n "\${AA_DIST}" ] && [ -f "\${AA_DIST}" ]; then
    OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/\${MODEL}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter\${ITERATION}"
    mkdir -p "\${OUTPUT_DIR}"

    python scripts/result_eval_uniform.py \
        --generated "\${AA_DIST}" \
        --baselines "/scratch/project_465002574/UNAAGI_benchmark_values/baselines/\${BASELINE}" \
        --total_num "\${NUM_SAMPLES}" \
        --output_dir "\${OUTPUT_DIR}"
fi
EOF
)

        POSTPROC_JOB=$(sbatch --parsable \
            --account=project_465002574 \
            --partition=small-g \
            --ntasks=1 \
            --cpus-per-task=4 \
            --mem=60G \
            --time=2-00:00:00 \
            -o /scratch/project_465002574/UAAG_logs/postproc_swift_%j.log \
            -e /scratch/project_465002574/UAAG_logs/postproc_swift_%j.log \
            --job-name="UAAG_swift_post_${assay_name}" \
            --dependency=afterok:${SAMPLING_JOB} \
            --export=ALL \
            --wrap "${postproc_wrap}" 2>&1)

        if [[ ${POSTPROC_JOB} =~ ^[0-9]+$ ]]; then
            echo "  ✓ Step 2: Post-analysis submitted: ${POSTPROC_JOB}"

            local cleanup_wrap
            cleanup_wrap=$(cat <<EOF
set -e
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:\$PATH"
mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

MODEL="${MODEL_NAME}"
NUM_SAMPLES="${NUM_SAMPLES}"
PROTEIN_ID="${protein_id}"
ITERATION=0

BASE_PATH="/scratch/project_465002574/ProteinGymSampling"
ARCHIVE_DIR="/scratch/project_465002574/UNAAGI_archives/\${MODEL}"
RUN_DIR="run\${MODEL}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter\${ITERATION}"
SAMPLES_DIR="\${BASE_PATH}/\${RUN_DIR}/Samples"
ARCHIVE_NAME="\${ARCHIVE_DIR}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter\${ITERATION}_mol_files.tar.gz"

mkdir -p "\${ARCHIVE_DIR}"

if [ ! -d "\${SAMPLES_DIR}" ]; then
    echo "Error: samples directory not found: \${SAMPLES_DIR}"
    exit 1
fi

cd "\${SAMPLES_DIR}"
MOL_COUNT=\$(find . \( -name "*.mol" -o -name "all_molecules.sdf" \) | wc -l)
if [ "\${MOL_COUNT}" -gt 0 ]; then
    find . \( -name "*.mol" -o -name "all_molecules.sdf" \) -print0 | tar -czf "\${ARCHIVE_NAME}" --null -T -
    find . -name "*.mol" -type f -delete
    find . -name "all_molecules.sdf" -type f -delete
    find . -type d -name "batch_*" -exec rm -rf {} + 2>/dev/null
    find . -type d -name "iter_*" -exec rm -rf {} + 2>/dev/null
    find . -type d -name "final" -exec rm -rf {} + 2>/dev/null
fi
EOF
)

            CLEANUP_JOB=$(sbatch --parsable \
                --account=project_465002574 \
                --partition=small-g \
                --ntasks=1 \
                --cpus-per-task=4 \
                --mem=60G \
                --time=2-00:00:00 \
                -o /scratch/project_465002574/UAAG_logs/cleanup_swift_%j.log \
                -e /scratch/project_465002574/UAAG_logs/cleanup_swift_%j.log \
                --job-name="UAAG_swift_clean_${assay_name}" \
                --dependency=afterok:${POSTPROC_JOB} \
                --export=ALL \
                --wrap "${cleanup_wrap}" 2>&1)

            if [[ ${CLEANUP_JOB} =~ ^[0-9]+$ ]]; then
                echo "  ✓ Step 3: Cleanup submitted: ${CLEANUP_JOB} (PoseBusters skipped)"
            else
                echo "  ✗ Cleanup submission failed: ${CLEANUP_JOB}"
            fi
        else
            echo "  ✗ Post-analysis submission failed: ${POSTPROC_JOB}"
        fi
    else
        echo "  ✗ Sampling job submission failed: ${SAMPLING_JOB}"
    fi

    echo ""
}

if [ "${ASSAY_NAME}" == "all" ]; then
    echo "============================================================================"
    echo "Submitting ALL assays in SWIFT mode"
    echo "Checkpoint: ${CKPT_PATH}"
    echo "Model: ${MODEL_NAME}"
    echo "Split count per assay: ${SPLIT_COUNT}"
    echo "============================================================================"
    echo ""

    for config_file in ${ASSAY_DIR}/*.txt; do
        submit_assay ${config_file}
        sleep 1
    done

    echo "============================================================================"
    echo "Submission complete!"
    echo "Check job status with: squeue -u \$USER"
    echo "============================================================================"
elif [ -f "${ASSAY_DIR}/${ASSAY_NAME}.txt" ]; then
    echo "============================================================================"
    echo "Submitting single assay in SWIFT mode: ${ASSAY_NAME}"
    echo "Split count: ${SPLIT_COUNT}"
    echo "============================================================================"
    echo ""

    submit_assay "${ASSAY_DIR}/${ASSAY_NAME}.txt"

    echo "============================================================================"
    echo "Check job status with: squeue -u \$USER"
    echo "============================================================================"
else
    echo "Error: Assay '${ASSAY_NAME}' not found!"
    echo ""
    echo "Available assays:"
    ls -1 ${ASSAY_DIR}/*.txt | xargs -n1 basename | sed 's/.txt$//' | nl
    exit 1
fi
