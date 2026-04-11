#!/bin/bash

# ============================================================================
# UAAG CTMC Tau-Leaping Pipeline Submission Script
# ============================================================================
# Same pipeline as submit_assay_job_swift.sh but uses DPM-Solver++ 2nd-order
# multistep sampler (every_k_step=25 → 20 NFE instead of 500, ~25x faster).
#
# Usage:
#   ./submit_assay_dpm_swift.sh <assay_name|all> <checkpoint_path> [num_samples] [model_name] [split_count] [every_k_step]
#
# Example:
#   ./submit_assay_dpm_swift.sh ENVZ_ECOLI \
#       /flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_40_gpu_UAAG_model_official_8_0202/last.ckpt \
#       100 dpm_pp_model 3 25
# ============================================================================

ASSAY_DIR="slurm_config/assays"
WORK_DIR="/flash/project_465002574/UAAG2_main"
LOG_DIR="/scratch/project_465002574/UAAG_logs"
DATA_INFO="/flash/project_465002574/UAAG2_main/data/statistic.pkl"
SAVE_DIR="/scratch/project_465002574/ProteinGymSampling"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <assay_name|all> <checkpoint_path> [num_samples] [model_name] [split_count] [every_k_step]"
    echo ""
    echo "Example:"
    echo "  $0 ENVZ_ECOLI /flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_40_gpu_UAAG_model_official_8_0202/last.ckpt 100 dpm_pp 3 25"
    exit 1
fi

ASSAY_NAME=$1
CKPT_PATH=$2
NUM_SAMPLES=${3:-100}
MODEL_NAME=${4:-$(basename "$(dirname "${CKPT_PATH}")")}
SPLIT_COUNT=${5:-3}
EVERY_K_STEP=${6:-25}   # 500/25 = 20 NFE with DPM-Solver++

if [ ! -f "${CKPT_PATH}" ]; then
    echo "Error: checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

echo "============================================================================"
echo "CTMC Assay Submission"
echo "  Checkpoint : ${CKPT_PATH}"
echo "  Model name : ${MODEL_NAME}"
echo "  every_k_step: ${EVERY_K_STEP}  (NFE = 500/${EVERY_K_STEP} = $((500/EVERY_K_STEP)))"
echo "  Samples/split: ${NUM_SAMPLES}"
echo "  Splits: ${SPLIT_COUNT}"
echo "============================================================================"
echo ""

submit_assay() {
    local config_file=$1
    local assay_name
    assay_name=$(basename "${config_file}" .txt)

    local protein_id baseline available_splits assay_split_count array_range
    protein_id=$(awk 'NR==2 {print $2}' "${config_file}")
    baseline=$(awk 'NR==2 {print $3}' "${config_file}")
    available_splits=$(awk 'NR>1 && $5==0 {print $4}' "${config_file}" | sort -n | uniq | wc -l | tr -d ' ')

    assay_split_count=${SPLIT_COUNT}
    if [ "${available_splits}" -lt "${assay_split_count}" ]; then
        assay_split_count=${available_splits}
    fi
    array_range="0-$((assay_split_count - 1))"

    echo "Submitting: ${assay_name}  (protein=${protein_id}, splits=${assay_split_count})"

    # ------------------------------------------------------------------
    # Step 1: Sampling via generate_ligand_dpm.py with DPM-Solver++
    # ------------------------------------------------------------------
    local sampling_wrap
    sampling_wrap=$(cat <<WRAP
set -e
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:\$PATH"
export PYTHONPATH="/flash/project_465002574/UAAG2_main/src:\$PYTHONPATH"
mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

echo "Branch: \$(git rev-parse --abbrev-ref HEAD)  commit: \$(git rev-parse --short HEAD)"

CONFIG_FILE="${config_file}"
CKPT_PATH="${CKPT_PATH}"
MODEL="${MODEL_NAME}"
NUM_SAMPLES="${NUM_SAMPLES}"
SPLIT_COUNT="${assay_split_count}"

ARRAY_ID=\${SLURM_ARRAY_TASK_ID}
PROTEIN_ID=\$(awk 'NR==2 {print \$2}' "\${CONFIG_FILE}")
SPLIT_LIST=\$(awk 'NR>1 && \$5==0 {print \$4}' "\${CONFIG_FILE}" | sort -n | uniq | head -n "\${SPLIT_COUNT}" | tr '\n' ' ')
SPLIT_INDEX=\$(echo "\${SPLIT_LIST}" | awk -v i=\${ARRAY_ID} '{print \$(i+1)}')

if [ -z "\${PROTEIN_ID}" ] || [ -z "\${SPLIT_INDEX}" ]; then
    echo "Error: failed to derive protein/split for task \${ARRAY_ID}"
    exit 1
fi

BENCHMARK_PATH="/scratch/project_465002574/UNAAGI_benchmarks/\${PROTEIN_ID}.pt"
RUN_ID="\${MODEL}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter0"

echo "Sampling: \${PROTEIN_ID}  split=\${SPLIT_INDEX}  every_k_step=${EVERY_K_STEP}  dpm_solver_pp=true"

python scripts/generate_ligand_dpm.py \
    --load-ckpt "\${CKPT_PATH}" \
    --id "\${RUN_ID}" \
    --benchmark-path "\${BENCHMARK_PATH}" \
    --split_index \${SPLIT_INDEX} \
    --virtual_node_size 15 \
    --num-samples "\${NUM_SAMPLES}" \
    --data_info_path "${DATA_INFO}" \
    --batch-size 8 \
    --num-workers 4 \
    --save-dir "${SAVE_DIR}" \
    --every-k-step ${EVERY_K_STEP} \
    --dpm-solver-pp \
    --ctmc
WRAP
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
        -o "${LOG_DIR}/ctmc_samp_${assay_name}_%A_%a.log" \
        -e "${LOG_DIR}/ctmc_samp_${assay_name}_%A_%a.log" \
        --job-name="CTMC_samp_${assay_name}" \
        --export=ALL \
        --wrap "${sampling_wrap}" 2>&1)

    if ! [[ ${SAMPLING_JOB} =~ ^[0-9]+$ ]]; then
        echo "  x Sampling submission failed: ${SAMPLING_JOB}"
        return
    fi
    echo "  + Step 1 Sampling: job ${SAMPLING_JOB} (${assay_split_count} tasks, ~$((500/EVERY_K_STEP)) NFE each)"

    # ------------------------------------------------------------------
    # Step 2: Post-analysis (aggregate mol files -> aa_distribution.csv)
    # ------------------------------------------------------------------
    local postproc_wrap
    postproc_wrap=$(cat <<WRAP
set -e
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:\$PATH"
export PYTHONPATH="/flash/project_465002574/UAAG2_main/src:\$PYTHONPATH"
cd "${WORK_DIR}"

MODEL="${MODEL_NAME}"
PROTEIN_ID="${protein_id}"
BASELINE="${baseline}"
NUM_SAMPLES="${NUM_SAMPLES}"
RUN_ID="\${MODEL}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter0"
SAMPLES_DIR="${SAVE_DIR}/run\${RUN_ID}/Samples"

if [ ! -d "\${SAMPLES_DIR}" ]; then
    echo "Error: samples dir not found: \${SAMPLES_DIR}"; exit 1
fi

python scripts/post_analysis.py --analysis_path "\${SAMPLES_DIR}"

AA_DIST="\${SAMPLES_DIR}/aa_distribution.csv"
if [ -f "\${AA_DIST}" ]; then
    OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/\${MODEL}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter0"
    mkdir -p "\${OUTPUT_DIR}"
    python scripts/result_eval_uniform.py \
        --generated "\${AA_DIST}" \
        --baselines "/scratch/project_465002574/UNAAGI_benchmark_values/baselines/${baseline}" \
        --total_num "\${NUM_SAMPLES}" \
        --output_dir "\${OUTPUT_DIR}"

    for UAA_NAME in PUMA CP2; do
        UAA_CSV="/scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/\${UAA_NAME}_reframe.csv"
        [ -f "\${UAA_CSV}" ] && python scripts/result_eval_uniform_uaa.py \
            --benchmark "\${UAA_CSV}" \
            --aa_output "\${AA_DIST}" \
            --output_dir "\${OUTPUT_DIR}/\${UAA_NAME}" \
            --total_num "\${NUM_SAMPLES}"
    done
fi
WRAP
)

    POSTPROC_JOB=$(sbatch --parsable \
        --account=project_465002574 \
        --partition=small-g \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=60G \
        --time=2-00:00:00 \
        -o "${LOG_DIR}/ctmc_post_${assay_name}_%j.log" \
        -e "${LOG_DIR}/ctmc_post_${assay_name}_%j.log" \
        --job-name="CTMC_post_${assay_name}" \
        --dependency=afterok:${SAMPLING_JOB} \
        --export=ALL \
        --wrap "${postproc_wrap}" 2>&1)

    if [[ ${POSTPROC_JOB} =~ ^[0-9]+$ ]]; then
        echo "  + Step 2 Post-analysis: job ${POSTPROC_JOB} (after ${SAMPLING_JOB})"
    else
        echo "  x Post-analysis submission failed: ${POSTPROC_JOB}"
        return
    fi

    # ------------------------------------------------------------------
    # Step 3: PoseBusters
    # ------------------------------------------------------------------
    local pb_wrap
    pb_wrap=$(cat <<WRAP
set -e
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:\$PATH"
export PYTHONPATH="/flash/project_465002574/UAAG2_main/src:\$PYTHONPATH"
cd "${WORK_DIR}"

MODEL="${MODEL_NAME}"
PROTEIN_ID="${protein_id}"
NUM_SAMPLES="${NUM_SAMPLES}"
RUN_ID="\${MODEL}/\${PROTEIN_ID}_\${MODEL}_\${NUM_SAMPLES}_iter0"
SAMPLES_DIR="${SAVE_DIR}/run\${RUN_ID}/Samples"

[ -d "\${SAMPLES_DIR}" ] || { echo "Samples dir missing: \${SAMPLES_DIR}"; exit 1; }

python scripts/evaluate_mol_samples.py \
    --input-dir "\${SAMPLES_DIR}" \
    --output "\${SAMPLES_DIR}/PoseBusterResults" \
    --temp-dir "/flash/project_465002574/temp_pb_dpm_${assay_name}_\${SLURM_JOB_ID}"
WRAP
)

    PB_JOB=$(sbatch --parsable \
        --account=project_465002574 \
        --partition=small-g \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=60G \
        --time=2-00:00:00 \
        -o "${LOG_DIR}/ctmc_pb_${assay_name}_%j.log" \
        -e "${LOG_DIR}/ctmc_pb_${assay_name}_%j.log" \
        --job-name="CTMC_pb_${assay_name}" \
        --dependency=afterok:${POSTPROC_JOB} \
        --export=ALL \
        --wrap "${pb_wrap}" 2>&1)

    if [[ ${PB_JOB} =~ ^[0-9]+$ ]]; then
        echo "  + Step 3 PoseBusters: job ${PB_JOB} (after ${POSTPROC_JOB})"
    else
        echo "  x PoseBusters submission failed: ${PB_JOB}"
    fi
    echo ""
}

# -------- dispatch --------
if [ "${ASSAY_NAME}" == "all" ]; then
    for config_file in ${ASSAY_DIR}/*.txt; do
        submit_assay "${config_file}"
        sleep 1
    done
elif [ -f "${ASSAY_DIR}/${ASSAY_NAME}.txt" ]; then
    submit_assay "${ASSAY_DIR}/${ASSAY_NAME}.txt"
else
    echo "Error: assay '${ASSAY_NAME}' not found. Available:"
    ls -1 ${ASSAY_DIR}/*.txt | xargs -n1 basename | sed 's/.txt//' | nl
    exit 1
fi

echo "============================================================================"
echo "Done. Monitor with: squeue -u \$USER"
echo "Logs: ${LOG_DIR}/dpm_*"
echo "============================================================================"

# -------- NCAA (PUMA / CP2) DPM submission --------
submit_uaa_dpm_job() {
    local uaa_name=$1
    local benchmark_pt=$2
    local benchmark_csv=$3
    local uaa_split_count=${4:-10}
    local uaa_array_range="0-$((uaa_split_count - 1))"

    echo "Submitting NCAA/UAA: ${uaa_name}  (splits=${uaa_split_count}, NFE=$((500/EVERY_K_STEP)))"

    local wrap
    wrap=$(cat <<WRAP
set -e
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:\$PATH"
export PYTHONPATH="/flash/project_465002574/UAAG2_main/src:\$PYTHONPATH"
mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"


UAA_NAME="${uaa_name}"
SPLIT_INDEX=\${SLURM_ARRAY_TASK_ID}
RUN_ID="${MODEL_NAME}/\${UAA_NAME}_${MODEL_NAME}_dpm_${NUM_SAMPLES}_split\${SPLIT_INDEX}"

python scripts/generate_ligand_dpm.py \
    --load-ckpt "${CKPT_PATH}" \
    --id "\${RUN_ID}" \
    --benchmark-path "${benchmark_pt}" \
    --split_index \${SPLIT_INDEX} \
    --total_partition ${uaa_split_count} \
    --virtual_node_size 15 \
    --num-samples "${NUM_SAMPLES}" \
    --data_info_path "${DATA_INFO}" \
    --batch-size 8 \
    --num-workers 4 \
    --save-dir "${SAVE_DIR}" \
    --every-k-step ${EVERY_K_STEP} \
    --dpm-solver-pp \
    --ctmc

SAMPLES_PATH="${SAVE_DIR}/run\${RUN_ID}/Samples"
OUTPUT_DIR="/scratch/project_465002574/UNAAGI_result/results/${MODEL_NAME}/\${UAA_NAME}_${MODEL_NAME}_dpm_${NUM_SAMPLES}_split\${SPLIT_INDEX}"
mkdir -p "\${OUTPUT_DIR}"

python scripts/post_analysis.py --analysis_path "\${SAMPLES_PATH}"

python scripts/result_eval_uniform_uaa.py \
    --benchmark "${benchmark_csv}" \
    --aa_output "\${SAMPLES_PATH}/aa_distribution.csv" \
    --output_dir "\${OUTPUT_DIR}" \
    --total_num "${NUM_SAMPLES}"
WRAP
)

    local job_id
    job_id=$(sbatch --parsable \
        --account=project_465002574 \
        --partition=standard-g \
        --ntasks=1 \
        --cpus-per-task=7 \
        --gpus-per-node=1 \
        --mem=60G \
        --time=2-00:00:00 \
        --array=${uaa_array_range} \
        -o "${LOG_DIR}/dpm_uaa_${uaa_name}_%A_%a.log" \
        -e "${LOG_DIR}/dpm_uaa_${uaa_name}_%A_%a.log" \
        --job-name="DPM_uaa_${uaa_name}" \
        --export=ALL \
        --wrap "${wrap}" 2>&1)

    if [[ ${job_id} =~ ^[0-9]+$ ]]; then
        echo "  + ${uaa_name}: job ${job_id} (${uaa_split_count} tasks)"
    else
        echo "  x ${uaa_name} submission failed: ${job_id}"
    fi
}
