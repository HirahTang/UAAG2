"""
run_train.py  — UAAG2 training entry point (dpm-solver-pp branch)

Merged from main branch run_train.py.  Supports training on any combination of:
  --use-lmdb      : pre-built LMDB shards (UAAG2Dataset, existing behaviour)
  --use-pdb       : on-the-fly PDB processing (UAAG2DatasetPDB)
  --use-pdbbind   : pre-built PDBBind LMDB (PDBBindDataset)
  --use-ncaa      : pre-built NCAA LMDB (PDBBindDataset, same class)

At least one source must be enabled.  Per-source sampling weights let you
up/down-weight each source without changing dataset sizes.
"""
import os
import sys
import warnings
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from argparse import ArgumentParser
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import ConcatDataset, RandomSampler

sys.path.insert(0, ".")
sys.path.insert(0, "..")

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

from uaag2.callbacks.ema import ExponentialMovingAverage
from uaag2.datasets.uaag_dataset import UAAG2DataModule, UAAG2Dataset, Dataset_Info
from uaag2.datasets.pdb_dataset import (
    UAAG2DatasetPDB,
    PDBBindDataset,
    CombinedPDBDataset,
)
from uaag2.equivariant_diffusion import Trainer


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _build_datasets(hparams):
    """Instantiate enabled dataset sources and return (all_data, source_weights)."""
    datasets = []
    weights_per_item = []   # parallel list of per-source float weights

    common_kw = dict(
        mask_rate=hparams.mask_rate,
        pocket_noise=hparams.pocket_noise,
        noise_scale=hparams.pocket_noise_scale,
        pocket_dropout_prob=hparams.pocket_dropout_prob,
        params=hparams,
    )

    # ---- 1. Pre-built LMDB (UAAG2Dataset) ----
    if hparams.use_lmdb:
        if not hparams.training_data:
            raise ValueError("--use-lmdb requires --training-data")
        ds = UAAG2Dataset(hparams.training_data, **common_kw)
        print(f"[dataset] LMDB:     {len(ds):>8,} graphs  "
              f"({len(ds.lmdb_paths)} shard(s))  weight={hparams.lmdb_weight}")
        datasets.append(ds)
        weights_per_item.extend([hparams.lmdb_weight] * len(ds))

    # ---- 2. On-the-fly PDB (UAAG2DatasetPDB) ----
    if hparams.use_pdb:
        if not hparams.pdb_dir:
            raise ValueError("--use-pdb requires --pdb-dir")
        ds = UAAG2DatasetPDB(
            pdb_dir=hparams.pdb_dir,
            latent_root_128=hparams.latent_root_128 or "",
            latent_root_20=hparams.latent_root_20 or "",
            pdb_fraction=hparams.pdb_fraction,
            max_pdb_files=hparams.pdb_max_files,
            **common_kw,
        )
        print(f"[dataset] PDB:      {len(ds):>8,} residues  weight={hparams.pdb_weight}")
        datasets.append(ds)
        weights_per_item.extend([hparams.pdb_weight] * len(ds))

    # ---- 3. PDBBind LMDB (PDBBindDataset) ----
    if hparams.use_pdbbind:
        if not hparams.pdbbind_lmdb:
            raise ValueError("--use-pdbbind requires --pdbbind-lmdb")
        ds = PDBBindDataset(hparams.pdbbind_lmdb, **common_kw)
        print(f"[dataset] PDBBind:  {len(ds):>8,} complexes  weight={hparams.pdbbind_weight}")
        datasets.append(ds)
        weights_per_item.extend([hparams.pdbbind_weight] * len(ds))

    # ---- 4. NCAA LMDB (PDBBindDataset, same class) ----
    if hparams.use_ncaa:
        if not hparams.ncaa_lmdb:
            raise ValueError("--use-ncaa requires --ncaa-lmdb")
        ds = PDBBindDataset(hparams.ncaa_lmdb, **common_kw)
        print(f"[dataset] NCAA:     {len(ds):>8,} structures  weight={hparams.ncaa_weight}")
        datasets.append(ds)
        weights_per_item.extend([hparams.ncaa_weight] * len(ds))

    if not datasets:
        raise ValueError(
            "No dataset source enabled. Pass at least one of: "
            "--use-lmdb, --use-pdb, --use-pdbbind, --use-ncaa"
        )

    all_data = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[dataset] TOTAL:    {len(all_data):>8,} samples across {len(datasets)} source(s)")
    return all_data, np.array(weights_per_item, dtype=float)


def _split(all_data, hparams):
    """Random train/val/test split.  Returns (train_data, val_data, test_data)."""
    n = len(all_data)
    n_test = int(hparams.test_size)
    n_val  = max(1, n - int(n * hparams.train_size) - n_test)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Dataset too small ({n}) for train_size={hparams.train_size} "
            f"and test_size={hparams.test_size}"
        )
    return torch.utils.data.random_split(all_data, [n_train, n_val, n_test])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(hparams):
    if hparams.use_protein_mpnn_context_128:
        hparams.context_mapping = True
        if hparams.num_context_features in (0, None):
            hparams.num_context_features = 128

    if hparams.context_mapping and hparams.num_context_features <= 0:
        raise ValueError(
            "When --context-mapping is enabled, --num-context-features must be > 0"
        )

    # ---- Callbacks & loggers ----
    ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hparams.save_dir, f"run{hparams.id}"),
        save_top_k=3,
        monitor="val/loss",
        save_last=True,
    )
    epoch_checkpoint_callback = None
    if hparams.save_every_epoch:
        epoch_checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(hparams.save_dir, f"run{hparams.id}"),
            filename="epoch{epoch:03d}",
            save_top_k=-1,
            every_n_epochs=1,
            save_last=False,
            monitor=None,
        )
    lr_logger = LearningRateMonitor()

    wandb_logger = WandbLogger(
        log_model="all",
        project="uaag2",
        name=f"run{hparams.id}",
    )
    tb_logger = TensorBoardLogger(
        os.path.join(hparams.save_dir, f"run{hparams.id}"),
        default_hp_metric=False,
    )
    if hparams.logger_type == "wandb":
        logger = wandb_logger
    elif hparams.logger_type == "tensorboard":
        logger = tb_logger
    else:
        raise ValueError(f"Unknown logger type: {hparams.logger_type}")

    # ---- Dataset_Info (for model construction) ----
    dataset_info = Dataset_Info(hparams, hparams.data_info_path)

    # ---- Build combined dataset ----
    print("\nBuilding datasets …")
    print(f"  pocket_noise={hparams.pocket_noise}  "
          f"mask_rate={hparams.mask_rate}  "
          f"noise_scale={hparams.pocket_noise_scale}  "
          f"pocket_dropout_prob={hparams.pocket_dropout_prob}")

    all_data, item_weights = _build_datasets(hparams)

    # ---- Train / val / test split ----
    train_data, val_data, test_data = _split(all_data, hparams)
    print(f"[split] train={len(train_data):,}  val={len(val_data):,}  test={len(test_data):,}")

    # ---- DataModule ----
    datamodule = UAAG2DataModule(hparams, train_data, val_data, test_data, sampler=None)

    # ---- Model ----
    model = Trainer(hparams=hparams, dataset_info=dataset_info)

    # ---- Trainer callbacks ----
    callbacks = [
        ema_callback,
        lr_logger,
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
    ]
    if epoch_checkpoint_callback is not None:
        callbacks.append(epoch_checkpoint_callback)
    if hparams.ema_decay == 1.0:
        callbacks = callbacks[1:]

    # ---- PL Trainer ----
    pl_trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else 1,
        strategy="ddp",
        plugins=LightningEnvironment(),
        num_nodes=hparams.num_nodes,
        logger=logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_batch,
        val_check_interval=hparams.eval_freq,
        gradient_clip_val=hparams.grad_clip_val,
        callbacks=callbacks,
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
        limit_train_batches=5000,
    )

    pl.seed_everything(seed=hparams.seed, workers=hparams.gpus > 1)

    # ---- Checkpoint loading ----
    ckpt_path = None
    if hparams.load_ckpt is not None:
        print("Loading from checkpoint …")
        ckpt_path = hparams.load_ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu")
        stored_lr = ckpt["optimizer_states"][0]["param_groups"][0]["lr"]
        if stored_lr != hparams.lr:
            print(f"  Updating lr {stored_lr} → {hparams.lr}")
            for pg in ckpt["optimizer_states"][0]["param_groups"]:
                pg["lr"] = hparams.lr
                pg["initial_lr"] = hparams.lr
            ckpt_path = os.path.join(
                os.path.dirname(hparams.load_ckpt),
                f"retraining_with_lr{hparams.lr}.ckpt",
            )
            if not os.path.exists(ckpt_path):
                torch.save(ckpt, ckpt_path)

    # ---- Save hparams ----
    hp_dir = os.path.join(hparams.save_dir, f"run{hparams.id}")
    os.makedirs(hp_dir, exist_ok=True)
    with open(os.path.join(hp_dir, "hparams.yaml"), "w") as fh:
        yaml.safe_dump(vars(hparams), fh)

    # ---- Fit ----
    pl_trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "3DcoordsAtomsBonds_0")
    parser = ArgumentParser()

    # ---- Logging ----
    parser.add_argument("--logger-type", default="wandb", type=str,
                        choices=["wandb", "tensorboard"])

    # ---- Dataset source switches ----
    parser.add_argument("--use-lmdb",    default=False, action="store_true",
                        help="Train on pre-built LMDB shards (UAAG2Dataset)")
    parser.add_argument("--use-pdb",     default=False, action="store_true",
                        help="Train on on-the-fly PDB processing (UAAG2DatasetPDB)")
    parser.add_argument("--use-pdbbind", default=False, action="store_true",
                        help="Train on PDBBind LMDB (PDBBindDataset)")
    parser.add_argument("--use-ncaa",    default=False, action="store_true",
                        help="Train on NCAA LMDB (PDBBindDataset)")

    # ---- Dataset paths ----
    parser.add_argument("--training-data", type=str, default="",
                        help="LMDB file or directory of .lmdb shards (--use-lmdb)")
    parser.add_argument("--pdb-dir", type=str, default="",
                        help="Directory of pre-cleaned .pdb files (--use-pdb)")
    parser.add_argument("--pdbbind-lmdb", type=str, default="",
                        help="Path to PDBBind.lmdb (--use-pdbbind)")
    parser.add_argument("--ncaa-lmdb", type=str, default="",
                        help="Path to NCAA.lmdb (--use-ncaa)")

    # ---- Per-source sampling weights ----
    parser.add_argument("--lmdb-weight",    default=1.0, type=float,
                        help="Sampling weight for LMDB source")
    parser.add_argument("--pdb-weight",     default=1.0, type=float,
                        help="Sampling weight for PDB source")
    parser.add_argument("--pdbbind-weight", default=1.0, type=float,
                        help="Sampling weight for PDBBind source")
    parser.add_argument("--ncaa-weight",    default=1.0, type=float,
                        help="Sampling weight for NCAA source")

    # ---- PDB-specific options ----
    parser.add_argument("--latent-root-128", type=str, default="",
                        help="ProteinMPNN latent root (dim 128) for UAAG2DatasetPDB")
    parser.add_argument("--latent-root-20",  type=str, default="",
                        help="ProteinMPNN latent root (dim 20) for UAAG2DatasetPDB")
    parser.add_argument("--pdb-fraction", type=float, default=1.0,
                        help="Fraction of PDB flat index to use (0<f<1 for subsets; default 1.0 = all)")
    parser.add_argument("--pdb-max-files", type=int, default=0,
                        help="Limit PDB index to first N files (0 = all; uses separate cache file)")

    # ---- Data info ----
    parser.add_argument("--data_info_path", type=str,
                        default="/flash/project_465002574/UAAG2_main/data/statistic.pkl")
    parser.add_argument("--save-every-epoch", default=False, action="store_true")
    parser.add_argument("--load-ckpt", default=None, type=str)
    parser.add_argument("--load-ckpt-from-pretrained", default=None, type=str)

    # ---- Files ----
    parser.add_argument("-s", "--save-dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--test-save-dir",  default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--train-size",     default=0.99, type=float)
    parser.add_argument("--val-size",       default=5000, type=float)
    parser.add_argument("--test-size",      default=100,  type=int)

    # ---- Model ----
    parser.add_argument("--dropout-prob",        default=0.3,    type=float)
    parser.add_argument("--virtual-node",        default=1,      type=int)
    parser.add_argument("--max-virtual-nodes",   default=11,     type=int)
    parser.add_argument("--sdim",                default=256,    type=int)
    parser.add_argument("--vdim",                default=64,     type=int)
    parser.add_argument("--latent_dim",          default=None,   type=int)
    parser.add_argument("--rbf-dim",             default=32,     type=int)
    parser.add_argument("--edim",                default=32,     type=int)
    parser.add_argument("--edge-mp",             default=False,  action="store_true")
    parser.add_argument("--vector-aggr",         default="mean", type=str)
    parser.add_argument("--num-layers",          default=7,      type=int)
    parser.add_argument("--fully-connected",     default=True,   action="store_true")
    parser.add_argument("--local-global-model",  default=False,  action="store_true")
    parser.add_argument("--local-edge-attrs",    default=False,  action="store_true")
    parser.add_argument("--use-cross-product",   default=False,  action="store_true")
    parser.add_argument("--cutoff-local",        default=7.0,    type=float)
    parser.add_argument("--cutoff-global",       default=10.0,   type=float)
    parser.add_argument("--use-pos-norm",        default=False,  action="store_true")
    parser.add_argument("--additional-feats",    default=True,   action="store_true")
    parser.add_argument("--use-qm-props",        default=False,  action="store_true")
    parser.add_argument("--build-mol-with-addfeats", default=False, action="store_true")

    # ---- Learning ----
    parser.add_argument("-b",  "--batch-size",          default=32,      type=int)
    parser.add_argument("-ib", "--inference-batch-size", default=32,     type=int)
    parser.add_argument("--gamma",                       default=0.975,  type=float)
    parser.add_argument("--grad-clip-val",               default=10.0,   type=float)
    parser.add_argument("--lr-scheduler",  default="reduce_on_plateau",
                        choices=["reduce_on_plateau", "cosine_annealing", "cyclic"])
    parser.add_argument("--optimizer",     default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr",            default=5e-4,  type=float)
    parser.add_argument("--lr-min",        default=5e-5,  type=float)
    parser.add_argument("--lr-step-size",  default=10000, type=int)
    parser.add_argument("--lr-frequency",  default=5,     type=int)
    parser.add_argument("--lr-patience",   default=20,    type=int)
    parser.add_argument("--lr-cooldown",   default=5,     type=int)
    parser.add_argument("--lr-factor",     default=0.75,  type=float)

    # ---- Diffusion ----
    parser.add_argument("--continuous",        default=False,  action="store_true")
    parser.add_argument("--noise-scheduler",   default="cosine",
                        choices=["linear","cosine","quad","sigmoid","adaptive","linear-time"])
    parser.add_argument("--eps-min",           default=1e-3,   type=float)
    parser.add_argument("--beta-min",          default=1e-4,   type=float)
    parser.add_argument("--beta-max",          default=2e-2,   type=float)
    parser.add_argument("--timesteps",         default=500,    type=int)
    parser.add_argument("--max-time",          default=None,   type=str)
    parser.add_argument("--lc-coords",         default=3.0,    type=float)
    parser.add_argument("--lc-atoms",          default=0.4,    type=float)
    parser.add_argument("--lc-bonds",          default=2.0,    type=float)
    parser.add_argument("--lc-charges",        default=1.0,    type=float)
    parser.add_argument("--lc-mulliken",       default=1.5,    type=float)
    parser.add_argument("--lc-wbo",            default=2.0,    type=float)
    parser.add_argument("--use-ligand-dataset-sizes", default=False, action="store_true")
    parser.add_argument("--loss-weighting",    default="snr_t",
                        choices=["snr_s_t","snr_t","exp_t","expt_t_half","uniform"])
    parser.add_argument("--snr-clamp-min",     default=0.05,   type=float)
    parser.add_argument("--snr-clamp-max",     default=1.50,   type=float)
    parser.add_argument("--ligand-pocket-interaction", default=False, action="store_true")
    parser.add_argument("--diffusion-pretraining",     default=False, action="store_true")
    parser.add_argument("--continuous-param",  default="data", type=str,
                        choices=["data","noise"])
    parser.add_argument("--atoms-categorical", default=True,   action="store_true")
    parser.add_argument("--bonds-categorical", default=True,   action="store_true")
    parser.add_argument("--atom-type-masking", default=True,   action="store_true")
    parser.add_argument("--use-absorbing-state", default=False, action="store_true")
    parser.add_argument("--num-bond-classes",  default=5,      type=int)
    parser.add_argument("--num-charge-classes", default=6,     type=int)

    # ---- Pocket / masking ----
    parser.add_argument("--pocket-noise",       default=False, action="store_true")
    parser.add_argument("--mask-rate",          default=0.5,   type=float,
                        help="0 = full mask (reconstruct all); 1 = no masking")
    parser.add_argument("--pocket-noise-scale", default=0.01,  type=float)
    parser.add_argument("--pocket-dropout-prob", default=0.0, type=float,
                        help="Probability of dropping all pocket atoms per sample (default 0 = off)")

    # ---- Bond / guidance ----
    parser.add_argument("--bond-guidance-model",       default=False, action="store_true")
    parser.add_argument("--bond-prediction",           default=False, action="store_true")
    parser.add_argument("--property-prediction",      default=False, action="store_true")
    parser.add_argument("--bond-model-guidance",       default=False, action="store_true")
    parser.add_argument("--energy-model-guidance",     default=False, action="store_true")
    parser.add_argument("--polarizabilty-model-guidance", default=False, action="store_true")
    parser.add_argument("--ckpt-bond-model",           default=None,  type=str)
    parser.add_argument("--ckpt-energy-model",         default=None,  type=str)
    parser.add_argument("--ckpt-polarizabilty-model",  default=None,  type=str)
    parser.add_argument("--guidance-scale",            default=1e-4,  type=float)

    # ---- Context ----
    parser.add_argument("--context-mapping",           default=False, action="store_true")
    parser.add_argument("--num-context-features",      default=0,     type=int)
    parser.add_argument("--use_protein_mpnn_context_128", default=False, action="store_true",
                        help="Enable 128-dim ProteinMPNN context conditioning")
    parser.add_argument("--properties-list",           default=[],    nargs="+", type=str)

    # ---- Latent ----
    parser.add_argument("--prior-beta",         default=1.0,   type=float)
    parser.add_argument("--sdim-latent",        default=256,   type=int)
    parser.add_argument("--vdim-latent",        default=64,    type=int)
    parser.add_argument("--latent-dim",         default=None,  type=int)
    parser.add_argument("--edim-latent",        default=32,    type=int)
    parser.add_argument("--num-layers-latent",  default=7,     type=int)
    parser.add_argument("--latent-layers",      default=7,     type=int)
    parser.add_argument("--latentmodel",        default="diffusion", type=str)
    parser.add_argument("--latent-detach",      default=False, action="store_true")

    # ---- General ----
    parser.add_argument("-i", "--id",           type=str,  default=0)
    parser.add_argument("--num_nodes",          default=40, type=int)
    parser.add_argument("-g", "--gpus",         default=1,  type=int)
    parser.add_argument("-e", "--num-epochs",   default=300, type=int)
    parser.add_argument("--eval-freq",          default=1.0, type=float)
    parser.add_argument("--test-interval",      default=5,  type=int)
    parser.add_argument("--precision",          default=32, type=int)
    parser.add_argument("--detect-anomaly",     default=False, action="store_true")
    parser.add_argument("--num-workers",        default=4,  type=int)
    parser.add_argument("--accum-batch",        default=1,  type=int)
    parser.add_argument("--max-num-neighbors",  default=128, type=int)
    parser.add_argument("--ema-decay",          default=0.9999, type=float)
    parser.add_argument("--weight-decay",       default=0.9999, type=float)
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--backprop-local",     default=False, action="store_true")
    parser.add_argument("--max-num-conformers", default=5,  type=int)
    parser.add_argument("--energy-training",    default=False, action="store_true")
    parser.add_argument("--property-training",  default=False, action="store_true")

    # ---- Sampling ----
    parser.add_argument("--num-test-graphs",    default=10000, type=int)
    parser.add_argument("--calculate-energy",   default=False, action="store_true")
    parser.add_argument("--save-xyz",           default=False, action="store_true")

    args = parser.parse_args()
    main(args)
