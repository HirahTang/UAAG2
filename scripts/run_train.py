import os
import sys
import wandb
import torch

sys.path.append('.')
sys.path.append('..')
import warnings
from argparse import ArgumentParser
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from uaag.data.uaag_dataset import UAAG2DataModule, UAAG2Dataset, UAAG2Dataset_sampling, Dataset_Info
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from uaag.callbacks.ema import ExponentialMovingAverage
from uaag.equivariant_diffusion import Trainer
from uaag.utils import load_data, load_model
from pytorch_lightning.plugins.environments import LightningEnvironment
import lmdb
from torch.utils.data import WeightedRandomSampler, RandomSampler

import pickle
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

# from eqgat_diff.experiments.data.distributions import DistributionProperty
# from eqgat_diff.experiments.hparams import add_arguments

def main(hparams):
    
    ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + f"/run{hparams.id}/",
        save_top_k=3,
        monitor="val/loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    wandb_logger = WandbLogger(
        log_model="all",
        project="uaag2",
        name=f"run{hparams.id}",
        )
    tb_logger = TensorBoardLogger(
            hparams.save_dir + f"/run{hparams.id}/", default_hp_metric=False
        )
    if hparams.logger_type == "wandb":
        logger = wandb_logger
    elif hparams.logger_type == "tensorboard":
        logger = tb_logger
    else:
        raise ValueError("Logger type not recognized")

    print("Loading DataModule")
    
    dataset_info = Dataset_Info(hparams, hparams.data_info_path)
    
    print("pocket noise: ", hparams.pocket_noise)
    print("mask rate: ", hparams.mask_rate)
    print("pocket noise scale: ", hparams.pocket_noise_scale)
    # lmdb_data_path = "/datasets/biochem/unaagi/unaagi_overfitting_train_v1.lmdb"
    lmdb_data_path = "/datasets/biochem/unaagi/unaagi_whole_v1.lmdb"
    all_data = UAAG2Dataset(lmdb_data_path,  mask_rate=hparams.mask_rate, pocket_noise=hparams.pocket_noise, noise_scale=hparams.pocket_noise_scale, params=hparams)
    test_data_setup = UAAG2Dataset(lmdb_data_path, params=hparams)
    # split all_data into train, val, test from all_data
    # from IPython import embed; embed()
    train_data, val_data, test_data = torch.utils.data.random_split(all_data, [int(len(all_data) * hparams.train_size), len(all_data) - int(len(all_data) * hparams.train_size) - int(hparams.test_size), int(hparams.test_size)])
    test_indices = test_data.indices
    test_data = torch.utils.data.Subset(test_data_setup, test_indices)
    # test_data
    # convert the UAAG2Dataset parameters in test_data
    
    with open("/datasets/biochem/unaagi/unaagi_whole_v1.metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    weights = []
    for i in range(len(train_data)):
        key = f"{i:08}".encode("ascii")
        source_name = metadata[key]
        if source_name in ["pdbbind_data.pt", "AACLBR.pt", "L_sidechain_data.pt"]:
            weights.append(hparams.pdbbind_weight)
        else:
            weights.append(1.0)
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    # sampler = RandomSampler(train_data)

    datamodule = UAAG2DataModule(hparams, train_data, val_data, test_data, sampler=sampler)
    
    model = Trainer(
        hparams=hparams,
        dataset_info=dataset_info,
    )
    
    
    strategy = "ddp" if hparams.gpus > 1 else "auto"
    # strategy = 'ddp_find_unused_parameters_true'
    callbacks = [
        ema_callback,
        lr_logger,
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
    ]

    if hparams.ema_decay == 1.0:
        callbacks = callbacks[1:]

    
    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else 1,
        strategy=strategy,
        plugins=LightningEnvironment(),
        num_nodes=1,
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
        limit_train_batches=30000,
    )
    
    pl.seed_everything(seed=hparams.seed, workers=hparams.gpus > 1)
    
    ckpt_path = None
    if hparams.load_ckpt is not None:
        print("Loading from checkpoint ...")
        

        ckpt_path = hparams.load_ckpt
        ckpt = torch.load(ckpt_path)
        if ckpt["optimizer_states"][0]["param_groups"][0]["lr"] != hparams.lr:
            print("Changing learning rate ...")
            ckpt["optimizer_states"][0]["param_groups"][0]["lr"] = hparams.lr
            ckpt["optimizer_states"][0]["param_groups"][0]["initial_lr"] = hparams.lr
            ckpt_path = (
                "lr" + "_" + str(hparams.lr) + "_" + os.path.basename(hparams.load_ckpt)
            )
            ckpt_path = os.path.join(
                os.path.dirname(hparams.load_ckpt),
                f"retraining_with_lr{hparams.lr}.ckpt",
            )
            if not os.path.exists(ckpt_path):
                torch.save(ckpt, ckpt_path)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path if hparams.load_ckpt is not None else None,
    )

    # save hparams
    hparams_path = os.path.join(hparams.save_dir, f"run{hparams.id}", "hparams.yaml")
    if not os.path.exists(os.path.dirname(hparams_path)):
        os.makedirs(os.path.dirname(hparams_path))
    with open(hparams_path, "w") as f:
        f.write(hparams.to_yaml())
    print(f"Saved hparams to {hparams_path}")

if __name__ == "__main__":
    
    DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "3DcoordsAtomsBonds_0")
    parser = ArgumentParser()
    # parser = add_arguments(parser)
    
    parser.add_argument("--logger-type", default="wandb", type=str)
    
    parser.add_argument('--dataset', type=str, default='drugs')
    
    parser.add_argument('--data_info_path', type=str, default="/home/qcx679/hantang/UAAG2/data/full_graph/statistic.pkl")
    
    # parser.add_argument(
    #     "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    # )  # keep first

    # Load from checkpoint
    parser.add_argument("--load-ckpt", default=None, type=str)
    parser.add_argument("--load-ckpt-from-pretrained", default=None, type=str)

    # DATA and FILES
    parser.add_argument("-s", "--save-dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--test-save-dir", default=DEFAULT_SAVE_DIR, type=str)

    parser.add_argument(
        "--dataset-root", default="----"
    )
    parser.add_argument("--use-adaptive-loader", default=True, action="store_true")
    parser.add_argument("--remove-hs", default=False, action="store_true")
    parser.add_argument("--select-train-subset", default=False, action="store_true")
    parser.add_argument("--train-size", default=0.99, type=float)
    parser.add_argument("--val-size", default=5000, type=float)
    parser.add_argument("--test-size", default=100, type=int)

    parser.add_argument("--dropout-prob", default=0.3, type=float)
    parser.add_argument("--virtual-node", default=1, type=int)
    parser.add_argument("--max-virtual-nodes", default=11, type=int)
    parser.add_argument("--pdbbind-weight", default=10.0, type=float)
    # LEARNING
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-ib", "--inference-batch-size", default=32, type=int)
    parser.add_argument("--gamma", default=0.975, type=float)
    parser.add_argument("--grad-clip-val", default=10.0, type=float)
    parser.add_argument(
        "--lr-scheduler",
        default="reduce_on_plateau",
        choices=["reduce_on_plateau", "cosine_annealing", "cyclic"],
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd"],
    )
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--lr-min", default=5e-5, type=float)
    parser.add_argument("--lr-step-size", default=10000, type=int)
    parser.add_argument("--lr-frequency", default=5, type=int)
    parser.add_argument("--lr-patience", default=20, type=int)
    parser.add_argument("--lr-cooldown", default=5, type=int)
    parser.add_argument("--lr-factor", default=0.75, type=float)

    # MODEL
    parser.add_argument("--sdim", default=256, type=int)
    parser.add_argument("--vdim", default=64, type=int)
    parser.add_argument("--latent_dim", default=None, type=int)
    parser.add_argument("--rbf-dim", default=32, type=int)
    parser.add_argument("--edim", default=32, type=int)
    parser.add_argument("--edge-mp", default=False, action="store_true")
    parser.add_argument("--vector-aggr", default="mean", type=str)
    parser.add_argument("--num-layers", default=7, type=int)
    parser.add_argument("--fully-connected", default=True, action="store_true")
    parser.add_argument("--local-global-model", default=False, action="store_true")
    parser.add_argument("--local-edge-attrs", default=False, action="store_true")
    parser.add_argument("--use-cross-product", default=False, action="store_true")
    parser.add_argument("--cutoff-local", default=7.0, type=float)
    parser.add_argument("--cutoff-global", default=10.0, type=float)
    parser.add_argument("--energy-training", default=False, action="store_true")
    parser.add_argument("--property-training", default=False, action="store_true")
    parser.add_argument(
        "--regression-property",
        default="polarizability",
        type=str,
        choices=[
            "dipole_norm",
            "total_energy",
            "HOMO-LUMO_gap",
            "dispersion",
            "atomisation_energy",
            "polarizability",
        ],
    )
    parser.add_argument("--energy-loss", default="l2", type=str, choices=["l2", "l1"])
    parser.add_argument("--use-pos-norm", default=False, action="store_true")

    # For Discrete: Include more features: (is_aromatic, is_in_ring, hybridization)
    parser.add_argument("--additional-feats", default=True, action="store_true")
    parser.add_argument("--use-qm-props", default=False, action="store_true")
    parser.add_argument("--build-mol-with-addfeats", default=False, action="store_true")

    # DIFFUSION
    parser.add_argument(
        "--continuous",
        default=False,
        action="store_true",
        help="If the diffusion process is applied on continuous time variable. Defaults to False",
    )
    parser.add_argument(
        "--noise-scheduler",
        default="cosine",
        choices=["linear", "cosine", "quad", "sigmoid", "adaptive", "linear-time"],
    )
    parser.add_argument("--eps-min", default=1e-3, type=float)
    parser.add_argument("--beta-min", default=1e-4, type=float)
    parser.add_argument("--beta-max", default=2e-2, type=float)
    parser.add_argument("--timesteps", default=500, type=int)
    parser.add_argument("--max-time", type=str, default=None)
    parser.add_argument("--lc-coords", default=3.0, type=float)
    parser.add_argument("--lc-atoms", default=0.4, type=float)
    parser.add_argument("--lc-bonds", default=2.0, type=float)
    parser.add_argument("--lc-charges", default=1.0, type=float)
    parser.add_argument("--lc-mulliken", default=1.5, type=float)
    parser.add_argument("--lc-wbo", default=2.0, type=float)

    parser.add_argument(
        "--use-ligand-dataset-sizes", default=False, action="store_true"
    )

    parser.add_argument(
        "--loss-weighting",
        default="snr_t",
        choices=["snr_s_t", "snr_t", "exp_t", "expt_t_half", "uniform"],
    )
    parser.add_argument("--snr-clamp-min", default=0.05, type=float)
    parser.add_argument("--snr-clamp-max", default=1.50, type=float)

    parser.add_argument(
        "--ligand-pocket-interaction", default=False, action="store_true"
    )
    parser.add_argument("--diffusion-pretraining", default=False, action="store_true")
    parser.add_argument(
        "--continuous-param", default="data", type=str, choices=["data", "noise"]
    )
    parser.add_argument("--atoms-categorical", default=True, action="store_true")
    parser.add_argument("--bonds-categorical", default=True, action="store_true")

    parser.add_argument("--atom-type-masking", default=True, action="store_true")
    parser.add_argument("--use-absorbing-state", default=False, action="store_true")

    parser.add_argument("--num-bond-classes", default=5, type=int)
    parser.add_argument("--num-charge-classes", default=6, type=int)

    parser.add_argument("--pocket-noise", default=False, action="store_true")
    parser.add_argument("--mask-rate", default=0.5, type=float, help="Mask rate, 0 for full mask and the model is reconstructing everything during training \
        , 1 for no masking and the model is reconstructing nothing during training")
    parser.add_argument("--pocket-noise-scale", default=0.01, type=float)
    
    # BOND PREDICTION AND GUIDANCE:
    parser.add_argument("--bond-guidance-model", default=False, action="store_true")
    parser.add_argument("--bond-prediction", default=False, action="store_true")
    parser.add_argument("--bond-model-guidance", default=False, action="store_true")
    parser.add_argument("--energy-model-guidance", default=False, action="store_true")
    parser.add_argument(
        "--polarizabilty-model-guidance", default=False, action="store_true"
    )
    parser.add_argument("--ckpt-bond-model", default=None, type=str)
    parser.add_argument("--ckpt-energy-model", default=None, type=str)
    parser.add_argument("--ckpt-polarizabilty-model", default=None, type=str)
    parser.add_argument("--guidance-scale", default=1.0e-4, type=float)

    # CONTEXT
    parser.add_argument("--context-mapping", default=False, action="store_true")
    parser.add_argument("--num-context-features", default=0, type=int)
    parser.add_argument("--properties-list", default=[], nargs="+", type=str)

    # PROPERTY PREDICTION
    parser.add_argument("--property-prediction", default=False, action="store_true")

    # LATENT
    parser.add_argument("--prior-beta", default=1.0, type=float)
    parser.add_argument("--sdim-latent", default=256, type=int)
    parser.add_argument("--vdim-latent", default=64, type=int)
    parser.add_argument("--latent-dim", default=None, type=int)
    parser.add_argument("--edim-latent", default=32, type=int)
    parser.add_argument("--num-layers-latent", default=7, type=int)
    parser.add_argument("--latent-layers", default=7, type=int)
    parser.add_argument("--latentmodel", default="diffusion", type=str)
    parser.add_argument("--latent-detach", default=False, action="store_true")

    # GENERAL
    parser.add_argument("-i", "--id", type=str, default=0)
    parser.add_argument("-g", "--gpus", default=1, type=int)
    parser.add_argument("-e", "--num-epochs", default=300, type=int)
    parser.add_argument("--eval-freq", default=1.0, type=float)
    parser.add_argument("--test-interval", default=5, type=int)
    parser.add_argument("-nh", "--no_h", default=False, action="store_true")
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--detect-anomaly", default=False, action="store_true")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--max-num-conformers",
        default=5,
        type=int,
        help="Maximum number of conformers per molecule. \
                            Defaults to 30. Set to -1 for all conformers available in database",
    )
    parser.add_argument("--accum-batch", default=1, type=int)
    parser.add_argument("--max-num-neighbors", default=128, type=int)
    parser.add_argument("--ema-decay", default=0.9999, type=float)
    parser.add_argument("--weight-decay", default=0.9999, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--backprop-local", default=False, action="store_true")

    # SAMPLING
    parser.add_argument("--num-test-graphs", default=10000, type=int)
    parser.add_argument("--calculate-energy", default=False, action="store_true")
    parser.add_argument("--save-xyz", default=False, action="store_true")
    parser.add_argument("--variational-sampling", default=False)
    
    args = parser.parse_args()
    
    main(args)
    

