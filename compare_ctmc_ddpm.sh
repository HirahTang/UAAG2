#!/bin/bash -l
#SBATCH --account=project_465002574
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --job-name=vis_ctmc_ddpm

module purge
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default

export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

python scripts/compare_ctmc_vs_ddpm.py --ddpm-results-dir /scratch/project_465002574/UNAAGI_result/results/ctmc_1k_5iter --output-dir /scratch/project_465002574/UNAAGI_result/figures/ctmc_prior_vs_ctmc_vanilla_p0203