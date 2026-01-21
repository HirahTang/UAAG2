import os
import pickle
import warnings

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import RandomSampler, WeightedRandomSampler

from uaag2.callbacks.ema import ExponentialMovingAverage
from uaag2.datasets.uaag_dataset import Dataset_Info, UAAG2DataModule, UAAG2Dataset
from uaag2.equivariant_diffusion import Trainer
from uaag2.logging_config import configure_file_logging, logger

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


def main(cfg: DictConfig):
    """Main training function using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing all training parameters
    """
    # Configure file logging to save logs alongside checkpoints
    log_dir = os.path.join(cfg.save_dir, "logs")
    configure_file_logging(log_dir)
    logger.info("Starting training run {}", cfg.id)
    logger.info("Configuration:\n{}", OmegaConf.to_yaml(cfg))

    ema_callback = ExponentialMovingAverage(decay=cfg.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        save_top_k=3,
        monitor="val/loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()

    # Initialize wandb run manually so we control it
    wandb_run = None
    if cfg.logger_type == "wandb":
        wandb_run = wandb.init(
            project="uaag2",
            name=f"run{cfg.id}",
        )

    wandb_logger = WandbLogger(
        experiment=wandb_run,  # Pass our run to the logger
        log_model=False,
    )
    tb_logger = TensorBoardLogger(cfg.save_dir, default_hp_metric=False)
    if cfg.logger_type == "wandb":
        pl_logger = wandb_logger
        # Log key hyperparameters to wandb
        wandb_logger.experiment.config.update(
            {
                "learning_rate": cfg.optimizer.lr,
                "batch_size": cfg.data.batch_size,
                "num_epochs": cfg.num_epochs,
                "num_layers": cfg.model.num_layers,
                "sdim": cfg.model.sdim,
                "vdim": cfg.model.vdim,
                "timesteps": cfg.diffusion.timesteps,
                "noise_scheduler": cfg.diffusion.noise_scheduler,
                "mask_rate": cfg.data.mask_rate,
                "seed": cfg.seed,
            }
        )
    elif cfg.logger_type == "tensorboard":
        pl_logger = tb_logger
    else:
        raise ValueError("Logger type not recognized")

    logger.info("Loading DataModule")

    dataset_info = Dataset_Info(cfg, cfg.data.data_info_path)

    logger.info("Pocket noise: {}", cfg.data.pocket_noise)
    logger.info("Mask rate: {}", cfg.data.mask_rate)
    logger.info("Pocket noise scale: {}", cfg.data.pocket_noise_scale)
    lmdb_data_path = cfg.data.training_data
    all_data = UAAG2Dataset(
        lmdb_data_path,
        mask_rate=cfg.data.mask_rate,
        pocket_noise=cfg.data.pocket_noise,
        noise_scale=cfg.data.pocket_noise_scale,
        params=cfg,
    )
    test_data_setup = UAAG2Dataset(lmdb_data_path, params=cfg)
    # split all_data into train, val, test from all_data
    train_data, val_data, test_data = torch.utils.data.random_split(
        all_data,
        [
            int(len(all_data) * cfg.data.train_size),
            len(all_data) - int(len(all_data) * cfg.data.train_size) - int(cfg.data.test_size),
            int(cfg.data.test_size),
        ],
    )
    test_indices = test_data.indices
    test_data = torch.utils.data.Subset(test_data_setup, test_indices)

    # Create sampler based on use_metadata_sampler flag
    if cfg.data.use_metadata_sampler and cfg.data.metadata_path is not None:
        logger.info("Using WeightedRandomSampler with metadata from: {}", cfg.data.metadata_path)
        with open(cfg.data.metadata_path, "rb") as f:
            metadata = pickle.load(f)
        weights = []
        for i in range(len(train_data)):
            key = f"{i:08}".encode("ascii")
            source_name = metadata[key]
            if source_name in ["pdbbind_data.pt", "AACLBR.pt", "L_sidechain_data.pt"]:
                weights.append(cfg.data.pdbbind_weight)
            else:
                weights.append(1.0)
        weights = np.array(weights)
        weights = weights / weights.sum()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    else:
        logger.info("Using RandomSampler (no metadata weighting)")
        sampler = RandomSampler(train_data)

    datamodule = UAAG2DataModule(cfg, train_data, val_data, test_data, sampler=sampler)

    # Create a flattened config for backward compatibility with Trainer
    # The Trainer class expects top-level parameters, but Hydra provides nested config
    flat_cfg = OmegaConf.create({
        **cfg,
        **cfg.model,
        **cfg.diffusion,
        **cfg.optimizer,
        # Keep nested versions for code that uses them
        "model": cfg.model,
        "data": cfg.data,
        "diffusion": cfg.diffusion,
        # Don't overwrite optimizer string with dict
        "optimizer": cfg.optimizer.name,
    })

    model = Trainer(
        hparams=flat_cfg,
        dataset_info=dataset_info,
    )

    strategy = "ddp" if cfg.gpus > 1 else "auto"
    callbacks = [
        ema_callback,
        lr_logger,
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
    ]

    if cfg.ema_decay == 1.0:
        callbacks = callbacks[1:]

    trainer = pl.Trainer(
        accelerator="gpu" if cfg.gpus else "cpu",
        devices=cfg.gpus if cfg.gpus else 1,
        strategy=strategy,
        plugins=LightningEnvironment(),
        num_nodes=1,
        logger=pl_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=cfg.accum_batch,
        val_check_interval=cfg.eval_freq,
        gradient_clip_val=cfg.optimizer.grad_clip_val,
        callbacks=callbacks,
        precision=cfg.precision,
        num_sanity_val_steps=2,
        max_epochs=cfg.num_epochs,
        detect_anomaly=cfg.detect_anomaly,
        limit_train_batches=30000,
    )

    pl.seed_everything(seed=cfg.seed, workers=cfg.gpus > 1)

    ckpt_path = None

    if cfg.load_ckpt is not None:
        logger.info("Loading from checkpoint: {}", cfg.load_ckpt)

        ckpt_path = cfg.load_ckpt
        ckpt = torch.load(ckpt_path)
        if ckpt["optimizer_states"][0]["param_groups"][0]["lr"] != cfg.optimizer.lr:
            logger.info("Changing learning rate to {}", cfg.optimizer.lr)
            ckpt["optimizer_states"][0]["param_groups"][0]["lr"] = cfg.optimizer.lr
            ckpt["optimizer_states"][0]["param_groups"][0]["initial_lr"] = cfg.optimizer.lr
            ckpt_path = os.path.join(
                os.path.dirname(cfg.load_ckpt),
                f"retraining_with_lr{cfg.optimizer.lr}.ckpt",
            )
            if not os.path.exists(ckpt_path):
                torch.save(ckpt, ckpt_path)

    # Save hydra config
    config_path = os.path.join(cfg.save_dir, "config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path if cfg.load_ckpt is not None else None,
    )

    # Log model artifact to wandb
    if cfg.logger_type == "wandb" and wandb_run is not None:
        ckpt_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or cfg.load_ckpt
        if not ckpt_path:
            ckpt_path = os.path.join(cfg.save_dir, "model.ckpt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            trainer.save_checkpoint(ckpt_path)

        artifact = wandb.Artifact(
            name="uaag2_model",
            type="model",
            metadata={"run_id": str(cfg.id), "epoch": trainer.current_epoch},
        )
        artifact.add_file(ckpt_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
        logger.info("Model artifact logged to wandb")


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def run(cfg: DictConfig) -> None:
    """Hydra entry point for training.

    Args:
        cfg: Hydra configuration loaded from YAML files
    """
    main(cfg)


if __name__ == "__main__":
    run()
