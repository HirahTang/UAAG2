# Switch to prior_condition branch
echo ""
echo "→ Switching to prior_condition branch..."
git checkout main


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "→ Setting up environment..."

WORK_DIR=/flash/project_465002574/UAAG2_main

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"
cd ${WORK_DIR}
python scripts/evaluate_mol_samples.py \
    --input-dir /scratch/project_465002574/UNAAGI_archives/UAAG_model/A0A247D711_LISMN_iter0_extracted/ASN_36 \
    --output /scratch/project_465002574/UNAAGI_archives/UAAG_model/A0A247D711_LISMN_iter0_extracted/ASN_36/PoseBusterResults \
    --temp-dir /flash/project_465002574/temp_sdf