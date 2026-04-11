#!/bin/bash
# submit_assay_ctmc_1k.sh  — CTMC 20 NFE, 1000 samples, 5 iterations
# One job per iteration (all positions in one run), then 1 aggregation job.
# Usage: ./submit_assay_ctmc_1k.sh <assay_name|all|uaa> [checkpoint_path]

WORK_DIR="/flash/project_465002574/UAAG2_main"
LOG_DIR="/scratch/project_465002574/UAAG_logs"
DATA_INFO="/flash/project_465002574/UAAG2_main/data/statistic.pkl"
SAVE_DIR="/scratch/project_465002574/ProteinGymSampling"
RESULT_DIR="/scratch/project_465002574/UNAAGI_result/results"
CKPT_DEFAULT="/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFull_mask_40_gpu_UAAG_model_official_8_0202/last.ckpt"
BENCH_DIR="/scratch/project_465002574/UNAAGI_benchmarks"
ASSAY_DIR="${WORK_DIR}/slurm_config/assays"

ASSAY_NAME=${1}
CKPT_PATH=${2:-${CKPT_DEFAULT}}
NUM_SAMPLES=1000
MODEL_NAME="ctmc_1k_5iter"
NUM_ITERS=5
EVERY_K_STEP=25

[ $# -lt 1 ] && { echo "Usage: $0 <assay|all|uaa> [ckpt]"; exit 1; }
[ -f "${CKPT_PATH}" ] || { echo "Checkpoint not found: ${CKPT_PATH}"; exit 1; }
mkdir -p "${LOG_DIR}"

echo "================================================================"
echo "CTMC 20 NFE  model=${MODEL_NAME}  samples=${NUM_SAMPLES}  iters=${NUM_ITERS}"
echo "================================================================"

# ------------------------------------------------------------------ #
# submit_assay: 5 sampling jobs (1 per iter) + 1 aggregation job     #
# ------------------------------------------------------------------ #
submit_assay() {
    local config_file=$1
    local protein_id baseline
    protein_id=$(awk 'NR==2 {print $2}' "${config_file}")
    baseline=$(awk 'NR==2 {print $3}' "${config_file}")
    local bench_pt="${BENCH_DIR}/${protein_id}.pt"
    [ -f "${bench_pt}" ] || { echo "  [SKIP] benchmark not found: ${bench_pt}"; return; }

    echo "Submitting: ${protein_id}"

    local ALL_SAMP_IDS=""
    for iter in $(seq 0 $((NUM_ITERS - 1))); do
        local RUN_ID="${MODEL_NAME}/${protein_id}_${MODEL_NAME}_${NUM_SAMPLES}_iter${iter}"
        local JID
        JID=$(sbatch --parsable \
            --account=project_465002574 --partition=standard-g \
            --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=60G --time=2-00:00:00 \
            -o "${LOG_DIR}/c1k_samp_${protein_id}_i${iter}_%j.log" \
            -e "${LOG_DIR}/c1k_samp_${protein_id}_i${iter}_%j.log" \
            --job-name="c1k_s_${protein_id:0:10}" --export=ALL \
            --wrap "set -e
module load LUMI; module load CrayEnv; module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH=/flash/project_465002574/unaagi_env/bin:\$PATH
export PYTHONPATH=${WORK_DIR}/src:\$PYTHONPATH
cd ${WORK_DIR}
python scripts/generate_ligand_dpm.py \
    --load-ckpt '${CKPT_PATH}' \
    --id '${RUN_ID}' \
    --benchmark-path '${bench_pt}' \
    --total_partition 1 --split_index 0 \
    --virtual_node_size 15 --num-samples ${NUM_SAMPLES} \
    --data_info_path '${DATA_INFO}' --batch-size 8 --num-workers 4 \
    --save-dir '${SAVE_DIR}' \
    --every-k-step ${EVERY_K_STEP} --dpm-solver-pp --ctmc" 2>&1)
        [[ ${JID} =~ ^[0-9]+$ ]] \
            && ALL_SAMP_IDS="${ALL_SAMP_IDS}:${JID}" \
            && echo "  + iter${iter}: job ${JID}" \
            || echo "  x iter${iter} FAILED: ${JID}"
    done

    [ -z "${ALL_SAMP_IDS}" ] && { echo "  x no sampling jobs submitted"; return; }

    # Aggregation: merge per-iter aa_distribution CSVs, run result_eval
    local AGGR_GLOB="${SAVE_DIR}/run${MODEL_NAME}/${protein_id}_${MODEL_NAME}_${NUM_SAMPLES}_iter*/aa_distribution_split0.csv"
    local AGGR_OUT="${RESULT_DIR}/${MODEL_NAME}/${protein_id}_${MODEL_NAME}_${NUM_SAMPLES}"
    local AGGR_JID
    AGGR_JID=$(sbatch --parsable \
        --account=project_465002574 --partition=small \
        --ntasks=1 --cpus-per-task=4 --mem=16G --time=1:00:00 \
        -o "${LOG_DIR}/c1k_aggr_${protein_id}_%j.log" \
        -e "${LOG_DIR}/c1k_aggr_${protein_id}_%j.log" \
        --job-name="c1k_a_${protein_id:0:10}" \
        --dependency="afterok${ALL_SAMP_IDS}" --export=ALL \
        --wrap "set -e
module load LUMI; module load CrayEnv; module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH=/flash/project_465002574/unaagi_env/bin:\$PATH
export PYTHONPATH=${WORK_DIR}/src:\$PYTHONPATH
cd ${WORK_DIR}
python scripts/aggregate_splits.py \
    --glob '${AGGR_GLOB}' \
    --output-dir '${AGGR_OUT}' \
    --baselines /scratch/project_465002574/UNAAGI_benchmark_values/baselines/${baseline} \
    --total-num $((NUM_SAMPLES * NUM_ITERS))" 2>&1)
    [[ ${AGGR_JID} =~ ^[0-9]+$ ]] \
        && echo "  + aggr: job ${AGGR_JID}" \
        || echo "  x aggr FAILED: ${AGGR_JID}"
    echo ""
}

# ------------------------------------------------------------------ #
# submit_uaa: same pattern but uses UAA benchmark + eval script       #
# ------------------------------------------------------------------ #
submit_uaa() {
    local uaa_name=$1 bench_pt=$2 bench_csv=$3
    echo "Submitting UAA: ${uaa_name}"

    local ALL_SAMP_IDS=""
    for iter in $(seq 0 $((NUM_ITERS - 1))); do
        local RUN_ID="${MODEL_NAME}/${uaa_name}_${MODEL_NAME}_${NUM_SAMPLES}_iter${iter}"
        local JID
        JID=$(sbatch --parsable \
            --account=project_465002574 --partition=standard-g \
            --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=60G --time=2-00:00:00 \
            -o "${LOG_DIR}/c1k_uaa_${uaa_name}_i${iter}_%j.log" \
            -e "${LOG_DIR}/c1k_uaa_${uaa_name}_i${iter}_%j.log" \
            --job-name="c1k_u_${uaa_name:0:10}" --export=ALL \
            --wrap "set -e
module load LUMI; module load CrayEnv; module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH=/flash/project_465002574/unaagi_env/bin:\$PATH
export PYTHONPATH=${WORK_DIR}/src:\$PYTHONPATH
cd ${WORK_DIR}
python scripts/generate_ligand_dpm.py \
    --load-ckpt '${CKPT_PATH}' \
    --id '${RUN_ID}' \
    --benchmark-path '${bench_pt}' \
    --total_partition 1 --split_index 0 \
    --virtual_node_size 15 --num-samples ${NUM_SAMPLES} \
    --data_info_path '${DATA_INFO}' --batch-size 8 --num-workers 4 \
    --save-dir '${SAVE_DIR}' \
    --every-k-step ${EVERY_K_STEP} --dpm-solver-pp --ctmc" 2>&1)
        [[ ${JID} =~ ^[0-9]+$ ]] \
            && ALL_SAMP_IDS="${ALL_SAMP_IDS}:${JID}" \
            && echo "  + iter${iter}: job ${JID}" \
            || echo "  x iter${iter} FAILED: ${JID}"
    done

    [ -z "${ALL_SAMP_IDS}" ] && return

    local AGGR_GLOB="${SAVE_DIR}/run${MODEL_NAME}/${uaa_name}_${MODEL_NAME}_${NUM_SAMPLES}_iter*/aa_distribution_split0.csv"
    local AGGR_OUT="${RESULT_DIR}/${MODEL_NAME}/${uaa_name}_${MODEL_NAME}_${NUM_SAMPLES}"
    local AGGR_JID
    AGGR_JID=$(sbatch --parsable \
        --account=project_465002574 --partition=small \
        --ntasks=1 --cpus-per-task=4 --mem=16G --time=1:00:00 \
        -o "${LOG_DIR}/c1k_uaa_aggr_${uaa_name}_%j.log" \
        -e "${LOG_DIR}/c1k_uaa_aggr_${uaa_name}_%j.log" \
        --job-name="c1k_ua_${uaa_name:0:10}" \
        --dependency="afterok${ALL_SAMP_IDS}" --export=ALL \
        --wrap "set -e
module load LUMI; module load CrayEnv; module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH=/flash/project_465002574/unaagi_env/bin:\$PATH
export PYTHONPATH=${WORK_DIR}/src:\$PYTHONPATH
cd ${WORK_DIR}
python scripts/aggregate_splits.py \
    --glob '${AGGR_GLOB}' \
    --output-dir '${AGGR_OUT}' \
    --uaa-benchmarks '${uaa_name}:${bench_csv}' \
    --total-num $((NUM_SAMPLES * NUM_ITERS))" 2>&1)
    [[ ${AGGR_JID} =~ ^[0-9]+$ ]] \
        && echo "  + aggr: job ${AGGR_JID}" \
        || echo "  x aggr FAILED: ${AGGR_JID}"
    echo ""
}

# ------------------------------------------------------------------ #
case "${ASSAY_NAME}" in
    all)
        for cfg in ${ASSAY_DIR}/*.txt; do submit_assay "${cfg}"; sleep 0.5; done
        submit_uaa PUMA \
            /scratch/project_465002574/UNAAGI_benchmarks/2roc_puma.pt \
            /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/PUMA_reframe.csv
        submit_uaa CP2 \
            /scratch/project_465002574/UNAAGI_benchmarks/5ly1_cp2.pt \
            /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/CP2_reframe.csv
        ;;
    uaa)
        submit_uaa PUMA \
            /scratch/project_465002574/UNAAGI_benchmarks/2roc_puma.pt \
            /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/PUMA_reframe.csv
        submit_uaa CP2 \
            /scratch/project_465002574/UNAAGI_benchmarks/5ly1_cp2.pt \
            /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/CP2_reframe.csv
        ;;
    *)
        cfg="${ASSAY_DIR}/${ASSAY_NAME}.txt"
        [ -f "${cfg}" ] && submit_assay "${cfg}" || { echo "Unknown assay: ${ASSAY_NAME}"; exit 1; }
        ;;
esac

echo "================================================================"
echo "Done. Monitor with: squeue -u \$USER"
echo "Logs: ${LOG_DIR}/c1k_*"
echo "================================================================"
