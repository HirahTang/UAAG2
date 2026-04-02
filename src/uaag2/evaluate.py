from __future__ import annotations

import os
import warnings

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from uaag2.datasets.uaag_dataset import Dataset_Info, UAAG2Dataset_sampling
from uaag2.equivariant_diffusion import Trainer
from uaag2.logging_config import configure_file_logging, logger

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


def main(cfg: DictConfig) -> None:
    """Run evaluation on a benchmark dataset using a trained model.

    Args:
        cfg: Configuration object containing model and evaluation parameters.
    """
    # Create save directory if it doesn't exist
    os.makedirs(cfg.save_dir, exist_ok=True)

    log_dir: str = os.path.join(cfg.save_dir, f"run{cfg.id}", "logs")
    configure_file_logging(log_dir)
    logger.info("Starting evaluation run {}", cfg.id)

    # Some evaluation specific params might need to be checked if they exist in config or added
    # Assuming they are passed via command line or exist in loaded config
    benchmark_path = getattr(cfg, "benchmark_path", None)
    if not benchmark_path:
        raise ValueError("benchmark_path must be specified in config or command line")

    logger.info("Loading data from: {}", benchmark_path)
    data_file = torch.load(benchmark_path, weights_only=False)

    # Split data into partitions based on split_index
    NUM_PARTITIONS: int = 10
    index: list[int] = list(range(len(data_file)))
    part_size: int = len(index) // NUM_PARTITIONS

    # Create partitions
    partitions: list[list[int]] = []
    for i in range(NUM_PARTITIONS - 1):
        partitions.append(index[i * part_size : (i + 1) * part_size])
    # Last partition gets any remaining elements
    partitions.append(index[(NUM_PARTITIONS - 1) * part_size :])

    # Select partition based on split_index
    split_index = getattr(cfg, "split_index", 0)
    if split_index < 0 or split_index >= NUM_PARTITIONS:
        raise ValueError(f"split_index must be between 0 and {NUM_PARTITIONS - 1}, got {split_index}")

    index = partitions[split_index]
    logger.info("Processing partition {}/{} with {} residues", split_index, NUM_PARTITIONS - 1, len(index))

    dataset_info = Dataset_Info(cfg, cfg.data.data_info_path)

    logger.info("Number of Residues: {}", len(index))

    # Evaluate params
    virtual_node_size = getattr(cfg, "virtual_node_size", 15)
    num_samples = getattr(cfg, "num_samples", 500)

    for graph_num in index:
        seq_position: int = int(data_file[graph_num].compound_id.split("_")[-3])
        seq_res: str = data_file[graph_num].compound_id.split("_")[-4]
        graph = data_file[graph_num]
        logger.info("Sampling for: {} {}", seq_res, seq_position)

        save_path: str = os.path.join(cfg.save_dir, "Samples", f"{seq_res}_{seq_position}")

        # Note: UAAG2Dataset_sampling expects hparams (namespace or config)
        dataset = UAAG2Dataset_sampling(
            graph,
            cfg,
            save_path,
            dataset_info,
            sample_size=virtual_node_size,
            sample_length=num_samples,
        )

        # Note: MPS is not used because torch_scatter doesn't support it
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=not torch.backends.mps.is_available(),
            shuffle=False,
        )

        # Load the model and checkpoint
        logger.info("Loading model from checkpoint: {}", cfg.load_ckpt)
        if hasattr(cfg, "load_ckpt") and cfg.load_ckpt:
            model = Trainer.load_from_checkpoint(
                cfg.load_ckpt,
                hparams=cfg,
                dataset_info=dataset_info,
            ).to(device)
        else:
            raise ValueError("load_ckpt must be provided for evaluation")

        model = model.eval()
        model.generate_ligand(dataloader, save_path=save_path, verbose=True)

    # Save the configuration
    config_path: str = os.path.join(cfg.save_dir, f"run{cfg.id}", "config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def run(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    run()
