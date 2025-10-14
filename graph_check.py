import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from tqdm import tqdm
import sys
sys.path.append('.')
sys.path.append('..')

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

import pytorch_lightning as pl
import torch.nn.functional as F

from uaag.data.uaag_dataset import UAAG2DataModule, UAAG2Dataset, UAAG2Dataset_sampling, Dataset_Info
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from argparse import ArgumentParser
from uaag.callbacks.ema import ExponentialMovingAverage

from torch_geometric.data import Dataset, DataLoader

from uaag.equivariant_diffusion import Trainer