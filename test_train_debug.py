"""Debug script to isolate where train.py crashes."""
import sys
print("Step 1: Basic imports starting...", flush=True)

import os
import pickle
import warnings
print("Step 2: Basic imports done", flush=True)

import hydra
print("Step 3: Hydra imported", flush=True)

import numpy as np
print("Step 4: NumPy imported", flush=True)

import pytorch_lightning as pl
print("Step 5: PyTorch Lightning imported", flush=True)

import torch
print("Step 6: PyTorch imported", flush=True)

import wandb
print("Step 7: WandB imported", flush=True)

from omegaconf import DictConfig, OmegaConf
print("Step 8: OmegaConf imported", flush=True)

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
print("Step 9: PL callbacks imported", flush=True)

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
print("Step 10: PL loggers imported", flush=True)

from pytorch_lightning.plugins.environments import LightningEnvironment
print("Step 11: PL plugins imported", flush=True)

from torch.utils.data import RandomSampler, WeightedRandomSampler
print("Step 12: PyTorch data utils imported", flush=True)

print("Step 13: About to import uaag2 modules...", flush=True)
from uaag2.callbacks.ema import ExponentialMovingAverage
print("Step 14: EMA callback imported", flush=True)

from uaag2.datasets.uaag_dataset import Dataset_Info, UAAG2DataModule, UAAG2Dataset
print("Step 15: Dataset modules imported", flush=True)

from uaag2.equivariant_diffusion import Trainer
print("Step 16: Trainer imported", flush=True)

from uaag2.logging_config import configure_file_logging, logger
print("Step 17: Logging config imported", flush=True)

print("\n✓ All imports successful!", flush=True)

@hydra.main(version_base=None, config_path="configs", config_name="train")
def test_hydra(cfg: DictConfig) -> None:
    """Test Hydra initialization."""
    print("\n✓ Hydra decorator executed!", flush=True)
    print("✓ Config loaded!", flush=True)
    print(f"Config keys: {list(cfg.keys())}", flush=True)

if __name__ == "__main__":
    print("\nStep 18: About to call Hydra main...", flush=True)
    test_hydra()
    print("\n✓ Script completed successfully!", flush=True)
