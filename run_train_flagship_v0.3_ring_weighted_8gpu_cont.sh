#!/bin/bash
#SBATCH --job-name=UAAG_v0.3_ring_w8gpu_cont
#SBATCH --account=project_465002988
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH -o /flash/project_465002574/UAAG2_main/logs/train_flagship_v0.3_ring_weighted_cont_%j.log
#SBATCH -e /flash/project_465002574/UAAG2_main/logs/train_flagship_v0.3_ring_weighted_cont_%j.log

module load LUMI
module load CrayEnv
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

export WORK_DIR=/flash/project_465002574/UAAG2_main
cd ${WORK_DIR}
mkdir -p logs

# MI250X stability + NCCL on Slingshot
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

RUN_ID=Flagship_v0.3_ring_weighted_8gpu_20260531_cont
PRIMARY_DIR=/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFlagship_v0.3_ring_weighted_8gpu_20260531

srun python scripts/run_train.py \
    --gpus 1 \
    --num_nodes 8 \
    --batch-size 4 \
    --accum-batch 4 \
    --num-epochs 100 \
    --logger-type wandb \
    --id ${RUN_ID} \
    --load-ckpt ${PRIMARY_DIR}/last.ckpt \
    --use-pdb \
    --use-pdbbind \
    --use-ncaa \
    --pdb-dir /flash/project_465002574/UAAG2_main/data/pdb_subset_1000_v0.11_symlinks \
    --pdbbind-lmdb /scratch/project_465002574/PDB/PDBBind.lmdb \
    --ncaa-lmdb /scratch/project_465002574/PDB/NCAA/NCAA.lmdb \
    --pdb-weight 1.0 \
    --pdbbind-weight 10.0 \
    --ncaa-weight 10.0 \
    --pdb-max-files 1000 \
    --pocket-dropout-prob 0.5 \
    --mask-rate 0.0 \
    --max-virtual-nodes 11 \
    --lr 0.0005 \
    --grad-clip-val 1.0 \
    --save-every-epoch \
    --data_info_path /flash/project_465002574/UAAG2_main/data/statistic.pkl
