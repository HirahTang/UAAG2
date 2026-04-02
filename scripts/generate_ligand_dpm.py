"""Sampling script for UAAG2 with optional DPM-Solver++ acceleration.

Drop-in replacement for scripts/generate_ligand.py using the uaag2 package.
Key additions over the original:
  --every-k-step   : subsample the 500-step chain (e.g. 25 → 20 NFE)
  --dpm-solver-pp  : use DPM-Solver++ 2nd-order multistep instead of DDPM

Usage (DPM-Solver++ with 20 NFE):
  python scripts/generate_ligand_dpm.py \
      --load-ckpt /path/to/last.ckpt \
      --id mymodel/ENVZ_ECOLI_100_iter0 \
      --benchmark-path /path/to/ENVZ_ECOLI.pt \
      --num-samples 100 \
      --split_index 0 \
      --every-k-step 25 \
      --dpm-solver-pp
"""
import argparse
import os
import sys
import yaml

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from uaag2.datasets.uaag_dataset import (
    Dataset_Info,
    UAAG2Dataset_sampling,
)
from uaag2.equivariant_diffusion import Trainer
from torch_geometric.loader import DataLoader


def _build_hparams(args):
    """Load saved checkpoint hparams and patch in the CLI overrides.

    This avoids having to replicate every hparam in argparse — we take the
    checkpoint's own hparams as the base and only patch the fields that
    differ between training and inference.
    """
    from argparse import Namespace
    ckpt = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
    saved = dict(ckpt.get("hyper_parameters", {}))

    # Inference overrides
    saved["load_ckpt"] = args.load_ckpt
    saved["load_ckpt_from_pretrained"] = None  # never re-load during eval
    saved["id"] = args.id
    saved["save_dir"] = args.save_dir
    saved["benchmark_path"] = args.benchmark_path
    saved["split_index"] = args.split_index
    saved["total_partition"] = args.total_partition
    saved["num_samples"] = args.num_samples
    saved["batch_size"] = args.batch_size
    saved["virtual_node_size"] = args.virtual_node_size
    saved["num_workers"] = args.num_workers
    saved["data_info_path"] = args.data_info_path
    saved["every_k_step"] = args.every_k_step
    saved["dpm_solver_pp"] = args.dpm_solver_pp
    saved["ddpm"] = args.ddpm
    saved["gpus"] = 1  # single-GPU inference

    return Namespace(**saved)


def main(args):
    hparams = _build_hparams(args)

    print("Loading data from:", args.benchmark_path)
    data_file = torch.load(args.benchmark_path, weights_only=False)

    NUM_PARTITIONS = args.total_partition
    index = list(range(len(data_file)))
    part_size = len(index) // NUM_PARTITIONS
    partitions = [index[i * part_size:(i + 1) * part_size] for i in range(NUM_PARTITIONS - 1)]
    partitions.append(index[(NUM_PARTITIONS - 1) * part_size:])

    if not (0 <= args.split_index < NUM_PARTITIONS):
        raise ValueError(f"split_index must be 0..{NUM_PARTITIONS - 1}, got {args.split_index}")

    index = partitions[args.split_index]
    print(f"Processing partition {args.split_index}/{NUM_PARTITIONS - 1} with {len(index)} residues")

    dataset_info = Dataset_Info(hparams, args.data_info_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model from checkpoint:", args.load_ckpt)
    model = Trainer.load_from_checkpoint(
        args.load_ckpt,
        hparams=hparams,
        dataset_info=dataset_info,
    ).to(device)
    model = model.eval()

    nfe = hparams.timesteps // args.every_k_step
    sampler = "DPM-Solver++" if args.dpm_solver_pp else ("DDPM" if args.ddpm else "DDIM")
    print(f"Sampler: {sampler}  every_k_step={args.every_k_step}  NFE≈{nfe}")

    print("Number of Residues to process:", len(index))
    for graph_num in index:
        seq_position = int(data_file[graph_num].compound_id.split("_")[-3])
        seq_res = data_file[graph_num].compound_id.split("_")[-4]
        graph = data_file[graph_num]
        print(f"Sampling for: {seq_res} {seq_position}")

        save_path = os.path.join("Samples", f"{seq_res}_{seq_position}")
        dataset = UAAG2Dataset_sampling(
            graph, hparams, save_path, dataset_info,
            sample_size=args.virtual_node_size,
            sample_length=args.num_samples,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        model.generate_ligand(
            dataloader,
            save_path=save_path,
            verbose=True,
            every_k_step=args.every_k_step,
            dpm_solver_pp=args.dpm_solver_pp,
            ddpm=args.ddpm,
        )

    # Save config
    config_path = os.path.join(args.save_dir, f"run{args.id}", "config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(vars(hparams), f)


if __name__ == "__main__":
    DEFAULT_SAVE_DIR = "/scratch/project_465002574/ProteinGymSampling"

    parser = argparse.ArgumentParser(description="UAAG2 sampling with DPM-Solver++")

    # Core paths
    parser.add_argument("--load-ckpt", required=True, type=str)
    parser.add_argument("--id", required=True, type=str)
    parser.add_argument("--benchmark-path", required=True, type=str)
    parser.add_argument("--data_info_path", default="/flash/project_465002574/UAAG2_main/data/statistic.pkl", type=str)
    parser.add_argument("-s", "--save-dir", default=DEFAULT_SAVE_DIR, type=str)

    # Sampling
    parser.add_argument("--num-samples", default=100, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--virtual_node_size", default=15, type=int)
    parser.add_argument("--split_index", default=0, type=int)
    parser.add_argument("--total_partition", default=10, type=int)
    parser.add_argument("--num-workers", default=4, type=int)

    # Sampler choice
    parser.add_argument("--every-k-step", default=25, type=int,
                        help="Step spacing: 500/25=20 NFE. Use 1 for full 500 steps.")
    parser.add_argument("--dpm-solver-pp", action="store_true",
                        help="Use DPM-Solver++ 2nd-order multistep (recommended with --every-k-step 25)")
    parser.add_argument("--ddpm", default=True, action="store_true",
                        help="Use DDPM (default when --dpm-solver-pp not set)")
    parser.add_argument("--eta-ddim", default=1.0, type=float)

    # Diffusion schedule (must match checkpoint)
    parser.add_argument("--timesteps", default=500, type=int)
    parser.add_argument("--noise-scheduler", default="cosine", type=str)
    parser.add_argument("--continuous-param", default="data", type=str)
    parser.add_argument("--beta-min", default=1e-4, type=float)
    parser.add_argument("--beta-max", default=2e-2, type=float)

    # Model dims (loaded from checkpoint; kept for hparams compatibility)
    parser.add_argument("--sdim", default=256, type=int)
    parser.add_argument("--vdim", default=64, type=int)
    parser.add_argument("--edim", default=32, type=int)
    parser.add_argument("--num-layers", default=7, type=int)
    parser.add_argument("--num-bond-classes", default=5, type=int)
    parser.add_argument("--num-charge-classes", default=6, type=int)
    parser.add_argument("--rbf-dim", default=32, type=int)
    parser.add_argument("--vector-aggr", default="mean", type=str)
    parser.add_argument("--cutoff-local", default=7.0, type=float)
    parser.add_argument("--cutoff-global", default=10.0, type=float)
    parser.add_argument("--latent_dim", default=None, type=int)
    parser.add_argument("--additional-feats", default=True, action="store_true")
    parser.add_argument("--atoms-categorical", default=True, action="store_true")
    parser.add_argument("--bonds-categorical", default=True, action="store_true")
    parser.add_argument("--atom-type-masking", default=True, action="store_true")
    parser.add_argument("--use-absorbing-state", default=False, action="store_true")
    parser.add_argument("--fully-connected", default=True, action="store_true")
    parser.add_argument("--ema-decay", default=0.9999, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--virtual_node", default=1, type=int)
    parser.add_argument("--dataset", default="drugs", type=str)

    args = parser.parse_args()
    main(args)
