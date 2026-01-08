#!/bin/bash
#SBATCH --job-name=UAAG2
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu,boomsma
#SBATCH --time=2-00:00:00
#SBATCH --exclude=hendrixgpu16fl,hendrixgpu19fl,hendrixgpu26fl
#SBATCH --output=Full_mask_5_virtual_node_mask_token_atomic_only_mask_diffusion_uniform_prior_1026_SER.out
nvidia-smi
echo "Job $SLURM_JOB_ID is running on node: $SLURMD_NODENAME"
echo "Hostname: $(hostname)"
# Make sure we're in the repo
cd /home/qcx679/hantang/UAAG2
source ~/.bashrc  # Ensure conda command is available
conda activate targetdiff
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qcx679/.conda/envs/targetdiff/lib

# Checkout the correct branch
git fetch origin
git checkout main

echo "Running on branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit hash:       $(git rev-parse HEAD)"

python scripts/run_train.py --logger-type wandb --batch-size 8 \
    --test-interval 5 --gpus 1 --mask-rate 0 --test-size 32 --train-size 0.99 \
    --id Full_mask_5_virtual_node_mask_token_atomic_only_mask_diffusion_1026_exclude_SER --max-virtual-nodes 5 --num-epochs 5000 \
    --training_data /datasets/biochem/unaagi/unaagi_exclude_SER.lmdb --metadata_path /datasets/biochem/unaagi/unaagi_exclude_SER.metadata.pkl