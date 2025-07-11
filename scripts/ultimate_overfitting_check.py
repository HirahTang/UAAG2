import os
import sys
import wandb
import torch

sys.path.append('.')
sys.path.append('..')
import warnings

from torch_geometric.data import Dataset, DataLoader, Data
from IPython import embed
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

class UAAG2OverfittingDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        graph,
        length: int = 1000,
        mask_rate: float = 0,
        pocket_noise: bool = False,
        noise_scale: float = 0.1,
        params=None,
    ):
        super(UAAG2OverfittingDataset, self).__init__()
        # self.statistics = Statistic()
        self.params = params
        self.pocket_noise = pocket_noise
        if self.pocket_noise:
            self.noise_scale = noise_scale
        
        
        
        # with self.env.begin() as txn:
        self.length = length
        # self.load_dataset()
        self.charge_emb = {
            -1: 0,
            0: 1,
            1: 2,
            2: 3,
        }
        self.mask_rate = mask_rate
        self.graph = self.preprocess(graph)
    def __len__(self):
        return self.length
    
        
    
    def pocket_centering(self, batch):
        # graph_data = self.data[idx]
        pos = batch.pos
        if batch.is_ligand.sum() == len(batch.is_ligand):
            pocket_mean = pos.mean(dim=0)
        else:
            pocket_pos = batch.pos[batch.is_ligand==0]
            pocket_mean = pocket_pos.mean(dim=0)
        pos = pos - pocket_mean
        batch.pos = pos
        return batch
    
    def preprocess(self, graph):
        graph_data = self.pocket_centering(graph)
        if not hasattr(graph_data, 'compound_id'):
            graph_data.compound_id = graph_data.componud_id
        if not hasattr(graph_data, 'edge_ligand'):
            graph_data.edge_ligand = torch.ones(graph_data.edge_attr.size(0))
        if not hasattr(graph_data, 'id'):
            graph_data.id = graph_data.compound_id
        graph_data.x = graph_data.x.float()
        graph_data.pos = graph_data.pos.float()
        graph_data.edge_attr = graph_data.edge_attr.float()
        graph_data.edge_index = graph_data.edge_index.long()
        # from IPython import embed; embed()
        
        charges_np = graph_data.charges.numpy()
        mapped_np = np.vectorize(self.charge_emb.get)(charges_np)
        charges = torch.from_numpy(mapped_np)
        
        graph_data.degree = graph_data.degree.float()
        graph_data.is_aromatic = graph_data.is_aromatic.float()
        graph_data.is_in_ring = graph_data.is_in_ring.float()
        graph_data.hybridization = graph_data.hybridization.float()
        graph_data.is_backbone = graph_data.is_backbone.float()
        graph_data.is_ligand = graph_data.is_ligand.float()
        
        batch_graph_data = Data(
            x=graph_data.x,
            pos=graph_data.pos,
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
            edge_ligand=torch.tensor(graph_data.edge_ligand).float(),
            charges=charges,
            degree=graph_data.degree,
            is_aromatic=graph_data.is_aromatic,
            is_in_ring=graph_data.is_in_ring,
            hybridization=graph_data.hybridization,
            is_backbone=graph_data.is_backbone,
            is_ligand=graph_data.is_ligand,
            ligand_size=torch.tensor(graph_data.is_ligand.sum() - graph_data.is_backbone.sum()).long(),
            id=graph_data.compound_id,
            ids=torch.tensor(range(len(graph_data.x))),
        )
        
        return batch_graph_data
        
    def __getitem__(self, idx):
        # deepcopy self.graph
        graph_data = self.graph.clone()
        # graph_data = self.graph
        assert isinstance(graph_data, Data), f"Expected torch_geometric.data.Data, got {type(graph)}"
        
        CoM = graph_data.pos[graph_data.is_ligand==1].mean(dim=0)
        
        reconstruct_mask = graph_data.is_ligand - graph_data.is_backbone
        new_backbone = graph_data.is_backbone[reconstruct_mask==1]
        new_backbone = torch.bernoulli(torch.ones_like(new_backbone) * self.mask_rate).float()
        graph_data.is_backbone[reconstruct_mask==1] = new_backbone
        reconstruct_size = (graph_data.is_ligand.sum() - graph_data.is_backbone.sum()).item()
        
        if reconstruct_size < self.params.max_virtual_nodes:
            sample_n = int(self.params.max_virtual_nodes - reconstruct_size)
        else:
            sample_n = np.random.randint(1, self.params.max_virtual_nodes)
        virtual_x = torch.ones(sample_n) * 8
        virtual_pos = torch.stack([CoM] * sample_n)
        
        virtual_charges = torch.ones(sample_n) * 3
        virtual_degree = torch.ones(sample_n) * 5
        virtual_is_aromatic = torch.ones(sample_n) * 2
        virtual_is_in_ring = torch.ones(sample_n) * 2
        virtual_hybridization = torch.ones(sample_n) * 4
        
        # append virtual_x to graph_data.x
        graph_data.x = torch.cat([graph_data.x, virtual_x])
        graph_data.pos = torch.cat([graph_data.pos, virtual_pos])
        graph_data.charges = torch.cat([graph_data.charges, virtual_charges])
        graph_data.degree = torch.cat([graph_data.degree, virtual_degree])
        graph_data.is_aromatic = torch.cat([graph_data.is_aromatic, virtual_is_aromatic])
        graph_data.is_in_ring = torch.cat([graph_data.is_in_ring, virtual_is_in_ring])
        graph_data.hybridization = torch.cat([graph_data.hybridization, virtual_hybridization])
        graph_data.is_backbone = torch.cat([graph_data.is_backbone, torch.zeros(sample_n)])
        graph_data.is_ligand = torch.cat([graph_data.is_ligand, torch.ones(sample_n)])
        
        virtual_new_id = torch.tensor(range(len(graph_data.x)))[-sample_n:]
        virtual_existed = torch.tensor(range(len(graph_data.x)))[:-sample_n]
        grid1, grid2 = torch.meshgrid(virtual_new_id, virtual_existed)
        grid1 = grid1.flatten()
        grid2 = grid2.flatten()
        # create the new edge index as a bidirectional graph
        new_edge_index = torch.stack([grid1, grid2])
        new_edge_index_reverse = torch.stack([grid2, grid1])
        new_edge_index = torch.cat([new_edge_index, new_edge_index_reverse], dim=1)

        edge_index_new = torch.cat([graph_data.edge_index, new_edge_index], dim=1)
        edge_attr_new = torch.cat([graph_data.edge_attr, torch.zeros(new_edge_index.size(1))])
        edge_ligand_new = torch.cat([torch.tensor(graph_data.edge_ligand), torch.zeros(new_edge_index.size(1))])
        
        grid1, grid2 = torch.meshgrid(virtual_new_id, virtual_new_id)
        mask = grid1 != grid2
        grid1 = grid1[mask]
        grid2 = grid2[mask]
        grid1 = grid1.flatten()
        grid2 = grid2.flatten()
        new_ligand_edge_index = torch.stack([grid1, grid2])
        edge_index_new = torch.cat([edge_index_new, new_ligand_edge_index], dim=1)
        edge_attr_new = torch.cat([edge_attr_new, torch.zeros(new_ligand_edge_index.size(1))])
        edge_ligand_new = torch.cat([edge_ligand_new, torch.ones(new_ligand_edge_index.size(1))]).float()
        
        graph_data.edge_index = edge_index_new
        graph_data.edge_attr = edge_attr_new
        graph_data.edge_ligand = edge_ligand_new
        
        graph_data.degree = graph_data.degree.float()
        graph_data.is_aromatic = graph_data.is_aromatic.float()
        graph_data.is_in_ring = graph_data.is_in_ring.float()
        graph_data.hybridization = graph_data.hybridization.float()
        graph_data.is_backbone = graph_data.is_backbone.float()
        graph_data.is_ligand = graph_data.is_ligand.float()
        
        batch_graph_data = Data(
            x=graph_data.x,
            pos=graph_data.pos,
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
            edge_ligand=torch.tensor(graph_data.edge_ligand).float(),
            charges=graph_data.charges,
            degree=graph_data.degree,
            is_aromatic=graph_data.is_aromatic,
            is_in_ring=graph_data.is_in_ring,
            hybridization=graph_data.hybridization,
            is_backbone=graph_data.is_backbone,
            is_ligand=graph_data.is_ligand,
            ligand_size=torch.tensor(graph_data.is_ligand.sum() - graph_data.is_backbone.sum()).long(),
            id=graph_data.id,
        )
        
        return batch_graph_data
        
def main(hparams):
    
    
    
    lr_logger = LearningRateMonitor()
    
    print("Loading DataModule")
    
    dataset_info = Dataset_Info(hparams, hparams.data_info_path)
    
    print("pocket noise: ", hparams.pocket_noise)
    print("mask rate: ", hparams.mask_rate)
    print("pocket noise scale: ", hparams.pocket_noise_scale)
    lmdb_data_path = "/datasets/biochem/unaagi/debug_test.lmdb"
    env = lmdb.open(
            lmdb_data_path,
            readonly=True,
            lock=False,
            subdir=False,
            readahead=False,
            meminit=False,
        )
    count = 0
    charge_emb = {
            -1: 0,
            0: 1,
            1: 2,
            2: 3,
        }
    # exp_id = hparams.id
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            graph = pickle.loads(value)
            ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)

            aa_identity = graph.compound_id.split("_")[2]
            if aa_identity in ['ARG', 'TYR', 'TRP', 'PHE', 'GLU']:
                count += 1
                if count > 10:
                    break
                print(graph)
                seq_position = int(graph.compound_id.split("_")[-3])
                seq_res = graph.compound_id.split("_")[-4]
                exp_id = f"{hparams.id}_{seq_res}_{seq_position}"
                print("Overfit Checking for: ", seq_res, seq_position)

                checkpoint_callback = ModelCheckpoint(
                    dirpath=hparams.save_dir + f"/run{exp_id}/",
                    save_top_k=3,
                    monitor="val/loss",
                    save_last=True,
                )
                ckpt_path = None      
                train_data = UAAG2OverfittingDataset(
                    graph=graph,
                    length=20000,
                    mask_rate=0,
                    pocket_noise=False,
                    params=hparams
                )
                
                # sample_dataset = UAAG2Dataset_sampling(graph, hparams, "sampling", dataset_info, sample_size=11, sample_length=100)
                
                # embed()
                
                val_data = UAAG2OverfittingDataset(
                    graph=graph,
                    length=1,
                    mask_rate=0,
                    pocket_noise=False,
                    params=hparams
                )
                
                test_data = UAAG2OverfittingDataset(
                    graph=graph,
                    length=1,
                    mask_rate=0,
                    pocket_noise=False,
                    params=hparams
                )
                
                sampler = RandomSampler(train_data)
                
                datamodule = UAAG2DataModule(hparams, train_data, val_data, test_data, sampler=sampler)
                
                model = Trainer(
                    hparams=hparams,
                    dataset_info=dataset_info,
                )
                wandb.init(project="uaag2", name=f"run{exp_id}", reinit=True)
                wandb_logger = WandbLogger(
                    log_model="all",
                    project="uaag2",
                    name=f"run{exp_id}",
                    )
                tb_logger = TensorBoardLogger(
                        hparams.save_dir + f"/run{exp_id}/", default_hp_metric=False
                    )
                if hparams.logger_type == "wandb":
                    logger = wandb_logger
                elif hparams.logger_type == "tensorboard":
                    logger = tb_logger
                else:
                    raise ValueError("Logger type not recognized")
                
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
                number_of_atom = 11
                
                sample_dataset = UAAG2Dataset_sampling(graph, hparams, "sampling", dataset_info, sample_size=number_of_atom, sample_length=100)
    
                sample_dataloader = DataLoader(
                    dataset=sample_dataset, 
                    batch_size=hparams.batch_size,
                    num_workers=hparams.num_workers,
                    pin_memory=True,
                    shuffle=False)
                
                model = model.eval()
                # save_path = os.path.join("sampling")
                model.generate_ligand(sample_dataloader, f"sampling_{exp_id}", verbose=True)
                wandb.finish()
    


if __name__ == '__main__':
    
    DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "OverFitting_Check")
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
    

