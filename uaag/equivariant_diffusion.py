import logging
from math import e
import os
from datetime import datetime
import json
import pickle
from platform import architecture
from re import L, S
import wandb
import sys

sys.path.append('.')
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from tqdm import tqdm

from uaag.e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from uaag.diffusion.continuous import DiscreteDDPM

from uaag.diffusion.categorical import CategoricalDiffusionKernel
from uaag.utils import load_model

from uaag.losses import DiffusionLoss


logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(
    logging.NullHandler()
)
logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(
    logging.NullHandler()
)

class Trainer(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        dataset_info,
        prop_dist=None,
        prop_norm=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0
        self.connected_components = 0.0
        
        
        
        self.qed = 0.0
        self.dataset_info = dataset_info
        self.prop_norm = prop_norm
        self.prop_dist = prop_dist
        # from IPython import embed; embed()
        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.bond_types.float()
        charge_types_distribution = dataset_info.charge_types.float()
        is_aromatic_distribution = dataset_info.is_aromatic.float()
        is_ring_distribution = dataset_info.is_ring.float()
        hybridization_distribution = dataset_info.hybridization.float()
        degree_distribution = dataset_info.degree.float()
        
        
        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())
        self.register_buffer("is_aromatic_prior", is_aromatic_distribution.clone())
        self.register_buffer("is_in_ring_prior", is_ring_distribution.clone())
        self.register_buffer("hybridization_prior", hybridization_distribution.clone())
        self.register_buffer("degree_prior", degree_distribution.clone())
        
        self.num_is_aromatic = self.num_is_in_ring = 2
        self.is_ligand = 1
        self.num_hybridization = len(hybridization_distribution)
        # self.num_hybridization = dataset_info.num_hybridization
        
        # self.hparams.num_atom_types = dataset_info.input_dims.X
        
        self.num_charge_classes = len(charge_types_distribution)
        self.num_atom_types = len(atom_types_distribution)
        
        self.num_degree = len(degree_distribution)

        self.num_atom_features = (
            self.num_atom_types
            + self.num_charge_classes
            + self.num_is_aromatic
            + self.num_is_in_ring
            + self.num_hybridization
            + self.num_degree
            + self.is_ligand
        )
        
        self.num_bond_classes = len(bond_types_distribution)
        
        if self.hparams.load_ckpt_from_pretrained is not None:
            print("Loading from pre-trained model checkpoint...")

            self.model = load_model(
            self.hparams.load_ckpt_from_pretrained, 
            self.num_atom_features, self.num_bond_classes,
            )
            # num_params = len(self.model.state_dict())
            # for i, param in enumerate(self.model.parameters()):
            #     if i < num_params // 2:
            #         param.requires_grad = False
        else:
            # from IPython import embed; embed()
            self.model = DenoisingEdgeNetwork(
                hn_dim=(hparams.sdim, hparams.vdim),
                num_layers=hparams.num_layers,
                latent_dim=None,
                use_cross_product=hparams.use_cross_product,
                num_atom_features=self.num_atom_features,
                num_bond_types=self.num_bond_classes,
                edge_dim=hparams.edim,
                cutoff_local=hparams.cutoff_local,
                vector_aggr=hparams.vector_aggr,
                fully_connected=hparams.fully_connected,
                local_global_model=hparams.local_global_model,
                recompute_edge_attributes=True,
                recompute_radius_graph=False,
                edge_mp=hparams.edge_mp,
                context_mapping=hparams.context_mapping,
                num_context_features=hparams.num_context_features,
                bond_prediction=hparams.bond_prediction,
                property_prediction=hparams.property_prediction,
                coords_param=hparams.continuous_param,
            )
        self.sde_pos = DiscreteDDPM(
                beta_min=hparams.beta_min,
                beta_max=hparams.beta_max,
                N=hparams.timesteps,
                scaled_reverse_posterior_sigma=True,
                schedule=self.hparams.noise_scheduler,
                nu=2.5,
                enforce_zero_terminal_snr=False,
                param=self.hparams.continuous_param,
            )
        self.sde_atom_charge = DiscreteDDPM(
            beta_min=hparams.beta_min,
            beta_max=hparams.beta_max,
            N=hparams.timesteps,
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=1,
            enforce_zero_terminal_snr=False,
        )
        self.sde_bonds = DiscreteDDPM(
            beta_min=hparams.beta_min,
            beta_max=hparams.beta_max,
            N=hparams.timesteps,
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=1.5,
            enforce_zero_terminal_snr=False,
        )
        
        self.cat_atoms = CategoricalDiffusionKernel(
            terminal_distribution=atom_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_distribution,
            alphas=self.sde_bonds.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )

        self.cat_aromatic = CategoricalDiffusionKernel(
            terminal_distribution=is_aromatic_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_is_aromatic=self.num_is_aromatic,
        )
        self.cat_ring = CategoricalDiffusionKernel(
            terminal_distribution=is_ring_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_is_in_ring=self.num_is_in_ring,
        )
        self.cat_hybridization = CategoricalDiffusionKernel(
            terminal_distribution=hybridization_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_hybridization=self.num_hybridization,
        )
        self.cat_degree = CategoricalDiffusionKernel(
            terminal_distribution=degree_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_degree=self.num_degree,
        )
        self.diffusion_loss = DiffusionLoss(
            modalities = [
                "coords",
                "atoms",
                "charges",
                "bonds",
                "ring",
                "aromatic",
                "hybridization",
                "degree",
            ],
            param=["data"] * 8,
        )
        
    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")
    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")
    def on_test_epoch_end(self):
        pass
    
    
    
    def _log(
        self,
        loss,
        coords_loss,
        atoms_loss,
        charges_loss,
        bonds_loss,
        ring_loss,
        aromatic_loss,
        hybridization_loss,
        degree_loss,
        batch_size,
        stage,
    ):
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=False,
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/atoms_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/charges_loss",
            charges_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/bonds_loss",
            bonds_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/ring_loss",
            ring_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/aromatic_loss",
            aromatic_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/hybridization_loss",
            hybridization_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/degree_loss",
            degree_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
    
    def step_fnc(self, batch, batch_idx, stage):
        
        batch_size = int(batch.batch.max()) + 1
        
        t = torch.randint(
            low=1,
            high=self.hparams.timesteps + 1,
            size=(batch_size,),
            dtype=torch.long,
            device=batch.x.device,
        )
        if self.hparams.loss_weighting == "snr_s_t":
            weights = self.sde_bonds.snr_s_t_weighting(
                s=t - 1, t=t, clamp_min=None, clamp_max=None
            ).to(batch.x.device)
        elif self.hparams.loss_weighting == "snr_t":
            weights = self.sde_bonds.snr_t_weighting(
                t=t, device=batch.x.device, clamp_min=0.05, clamp_max=5.0
            )
        elif self.hparams.loss_weighting == "exp_t":
            weights = self.sde_atom_charge.exp_t_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "exp_t_half":
            weights = self.sde_atom_charge.exp_t_half_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "uniform":
            weights = None
        
        pocket_mask = (1 - batch.is_ligand + batch.is_backbone).long()
        ligand_mask = (batch.is_ligand - batch.is_backbone).long()
        
        out_dict = self(batch=batch, t=t)
        
        true_data = {
            "coords": out_dict["coords_true"]
            if self.hparams.continuous_param == "data"
            else out_dict["coords_noise_true"],
            "atoms": out_dict["atoms_true"],
            "charges": out_dict["charges_true"],
            "bonds": out_dict["bonds_true"],
            "ring": out_dict["ring_true"],
            "aromatic": out_dict["aromatic_true"],
            "hybridization": out_dict["hybridization_true"],
            "degree": out_dict["degree_true"],
        }
        
        coords_pred = out_dict["coords_pred"]
        atoms_pred = out_dict["atoms_pred"]
        
        (
            atoms_pred,
            charges_pred,
            ring_pred,
            aromatic_pred,
            hybridization_pred,
            degree_pred,
            _
        ) = atoms_pred.split(
            [
                self.num_atom_types,
                self.num_charge_classes,
                self.num_is_in_ring,
                self.num_is_aromatic,
                self.num_hybridization,
                self.num_degree,
                self.is_ligand,
            ],
            dim=-1,
        )
        
        edges_pred = out_dict["bonds_pred"]
        
        pred_data = {
            "coords": coords_pred,
            "atoms": atoms_pred,
            "charges": charges_pred,
            "bonds": edges_pred,
            "ring": ring_pred,
            "aromatic": aromatic_pred,
            "hybridization": hybridization_pred,
            "degree": degree_pred,
        }
        # from IPython import embed; embed()
        loss = self.diffusion_loss(
            pred_data=pred_data,
            true_data=true_data,
            batch=batch.batch[ligand_mask==1],
            bond_aggregation_index=out_dict["edge_batch"],
            weights=weights,
            batch_size=batch_size,
        )
        
        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
            + 0.5 * loss["ring"]
            + 0.7 * loss["aromatic"]
            + 1.0 * loss["hybridization"]
            + 1.0 * loss["degree"]
        )
        
        if torch.any(final_loss.isnan()):
            final_loss = final_loss[~final_loss.isnan()]
            print(f"Detected NaNs. Terminating training at epoch {self.current_epoch}")
            exit()

        self._log(
            final_loss,
            loss["coords"],
            loss["atoms"],
            loss["charges"],
            loss["bonds"],
            loss["ring"],
            loss["aromatic"],
            loss["hybridization"],
            loss['degree'],
            batch_size,
            stage,
        )
        
        return final_loss
    
    def forward(self, batch: Batch, t: Tensor):
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        
        n = batch.num_nodes
        bs = batch.batch.max() + 1
        
        ring_feat = batch.is_in_ring
        aromatic_feat = batch.is_aromatic
        hybridization_feat = batch.hybridization
        degree_feat = batch.degree
        
        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)
        
        pocket_mask = (1 - batch.is_ligand + batch.is_backbone).long()
        ligand_mask = (batch.is_ligand - batch.is_backbone).long()
        # pocket noise?
        
        noise_coords_true, pos_perturbed = self.sde_pos.sample_pos(
            t,
            pos,
            data_batch,
            remove_mean=False,
        )
        # substiute the noised pocket atoms with the original ones, vice versa for all the features
        # take noise coords true by the ligand_mask==1
        
        pos_perturbed = pos_perturbed * ligand_mask.unsqueeze(1) + pos * pocket_mask.unsqueeze(1)
        noise_coords_true = noise_coords_true[ligand_mask==1]
        pos_true = pos[ligand_mask==1]
        
        atom_types, atom_types_perturbed = self.cat_atoms.sample_categorical(
            t,
            atom_types,
            data_batch,
            self.dataset_info,
            num_classes=self.num_atom_types,
            type="atoms",
        )
        
        atom_types_perturbed = atom_types_perturbed * ligand_mask.unsqueeze(1) + atom_types * pocket_mask.unsqueeze(1)
        atom_types = atom_types[ligand_mask==1]
        
        charges, charges_perturbed = self.cat_charges.sample_categorical(
            t,
            charges,
            data_batch,
            self.dataset_info,
            num_classes=self.num_charge_classes,
            type="charges",
        )
        
        charges_perturbed = charges_perturbed * ligand_mask.unsqueeze(1) + charges * pocket_mask.unsqueeze(1)
        charges = charges[ligand_mask==1]

        edge_attr_global_perturbed, edge_attr_global_original = self.cat_bonds.sample_edges_categorical(
                t, bond_edge_index, bond_edge_attr, data_batch
            )
        
        edge_attr_global_perturbed = edge_attr_global_perturbed.float()
        edge_attr_global_original = edge_attr_global_original.float()
        edge_attr_global_perturbed = edge_attr_global_perturbed * batch.edge_ligand.unsqueeze(1) + edge_attr_global_original * (1-batch.edge_ligand).unsqueeze(1)
        edge_attr_global_original = edge_attr_global_original[batch.edge_ligand==1]
        
        ring_feat, ring_feat_perturbed = self.cat_ring.sample_categorical(
            t,
            ring_feat,
            data_batch,
            self.dataset_info,
            num_classes=self.num_is_in_ring,
            type="ring",
        )
        ring_feat_perturbed = ring_feat_perturbed * ligand_mask.unsqueeze(1) + ring_feat * pocket_mask.unsqueeze(1)
        ring_feat = ring_feat[ligand_mask==1]
        
        
        (
            aromatic_feat,
            aromatic_feat_perturbed,
        ) = self.cat_aromatic.sample_categorical(
            t,
            aromatic_feat,
            data_batch,
            self.dataset_info,
            num_classes=self.num_is_aromatic,
            type="aromatic",
        )
        aromatic_feat_perturbed = aromatic_feat_perturbed * ligand_mask.unsqueeze(1) + aromatic_feat * pocket_mask.unsqueeze(1)
        aromatic_feat = aromatic_feat[ligand_mask==1]       
       
       
       
        (
            hybridization_feat,
            hybridization_feat_perturbed,
        ) = self.cat_hybridization.sample_categorical(
            t,
            hybridization_feat,
            data_batch,
            self.dataset_info,
            num_classes=self.num_hybridization,
            type="hybridization",
        )
        hybridization_feat_perturbed = hybridization_feat_perturbed * ligand_mask.unsqueeze(1) + hybridization_feat * pocket_mask.unsqueeze(1)
        hybridization_feat = hybridization_feat[ligand_mask==1]
        
        
        (
            degree_feat,
            degree_feat_perturbed,
        ) = self.cat_degree.sample_categorical(
            t,
            degree_feat,
            data_batch,
            self.dataset_info,
            num_classes=self.num_degree,
            type="degree",
        )
        degree_feat_perturbed = degree_feat_perturbed * ligand_mask.unsqueeze(1) + degree_feat * pocket_mask.unsqueeze(1)
        degree_feat = degree_feat[ligand_mask==1]
        
        
        batch_is_ligand = batch.is_ligand.unsqueeze(1)
        atom_feats_in_perturbed = torch.cat(
            [
                atom_types_perturbed,
                charges_perturbed,
                ring_feat_perturbed,
                aromatic_feat_perturbed,
                hybridization_feat_perturbed,
                degree_feat_perturbed,
                batch_is_ligand,
                
            ],
            dim=-1,
        )
        
        
        # from IPython import embed; embed()
        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=bond_edge_index,
            edge_attr_global=edge_attr_global_perturbed,
            batch=data_batch,
            batch_edge_global=batch.edge_ligand.long(),
            context=None,
            pocket_mask=pocket_mask.unsqueeze(1),
            edge_mask=batch.edge_ligand.long(),
            batch_lig=batch.batch[pocket_mask==0],
        )
        # from IPython import embed; embed()
        
        
        out["coords_perturbed"] = pos_perturbed[ligand_mask==1]
        out["atoms_perturbed"] = atom_types_perturbed[ligand_mask==1]
        out["charges_perturbed"] = charges_perturbed[ligand_mask==1]
        out["bonds_perturbed"] = edge_attr_global_perturbed[batch.edge_ligand==1]
        out["ring_perturbed"] = ring_feat_perturbed[ligand_mask==1]
        out["aromatic_perturbed"] = aromatic_feat_perturbed[ligand_mask==1]
        out["hybridization_perturbed"] = hybridization_feat_perturbed[ligand_mask==1]
        out["degree_perturbed"] = degree_feat_perturbed[ligand_mask==1]

        out["coords_true"] = pos_true
        out["coords_noise_true"] = noise_coords_true
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global_original
        out["charges_true"] = charges.argmax(dim=-1)
        out["ring_true"] = ring_feat.argmax(dim=-1)
        out["aromatic_true"] = aromatic_feat.argmax(dim=-1)
        out["hybridization_true"] = hybridization_feat.argmax(dim=-1)
        out["degree_true"] = degree_feat.argmax(dim=-1)
        out["bond_aggregation_index"] = bond_edge_index[1][batch.edge_ligand==1]
        out['edge_batch'] = batch.batch[bond_edge_index[0]][batch.edge_ligand==1]
        # from IPython import embed; embed()
        return out
    
    # def on_validation_epoch_end(self):
        
    #     final_res = self.run_evaluation(
    #         step=self.i,
    #             device="cuda" if self.hparams.gpus > 1 else "cpu",
    #             dataset_info=self.dataset_info,
    #             ngraphs=64,
    #             bs=self.hparams.inference_batch_size,
    #             verbose=True,
    #             inner_verbose=False,
    #             eta_ddim=1.0,
    #             ddpm=True,
    #             every_k_step=1,
    #     )
           
         
    
    # @torch.no_grad()
    # def generate_graphs(
    #     self,
    #     num_graphs: int,
    #     empirical_distribution_num_nodes: torch.Tensor,
    #     device: torch.device,
    #     verbose=False,
    #     save_traj=False,
    #     ddpm: bool = True,
    #     eta_ddim: float = 1.0,
    #     every_k_step: int = 1,
    # ):
    #     pass
    
    # @torch.no_grad()
    # def run_evaluation(
    #     self,
    #     step: int,
    #     dataset_info,
    #     ngraphs: int = 4000,
    #     bs: int = 500,
    #     save_dir: str = None,
    #     return_molecules: bool = False,
    #     verbose: bool = False,
    #     inner_verbose=False,
    #     ddpm: bool = True,
    #     eta_ddim: float = 1.0,
    #     every_k_step: int = 1,
    #     run_test_eval: bool = False,
    #     save_traj: bool = False,
    #     device: str = "cpu",
    #     **kwargs,
    # ):
    #     b = ngraphs // bs
    #     l = [bs] * b
    #     if sum(l) != ngraphs:
    #         l.append(ngraphs - sum(l))
    #     assert sum(l) == ngraphs
        
    #     molecule_list = []
    #     start = datetime.now()
        
    #     if verbose:
    #         if self.local_rank == 0:
    #             print(f"Creating {ngraphs} graphs in {l} batches")
                
    #     for _, num_graphs in enumerate(l):
    #         (
    #             pos_splits,
    #             atom_types_integer_split,
    #             charge_types_integer_split,
    #             aromatic_feat_integer_split,
    #             hybridization_feat_integer_split,
    #             edge_types,
    #             edge_index_global,
    #             batch_num_nodes,
    #             trajs,
    #             context_split,
    #         ) = self.generate_graphs(
    #             num_graphs=num_graphs,
    #             verbose=inner_verbose,
    #             device=self.device,
    #             empirical_distribution_num_nodes=self.empirical_num_nodes,
    #             save_traj=save_traj,
    #             ddpm=ddpm,
    #             eta_ddim=eta_ddim,
    #             every_k_step=every_k_step,
    #         )
    
    
    # def reverse_sampling(
    #     self, 
    #     num_graphs: int,
    #     empirical_distribution_num_nodes: Tensor,
    #     device: torch.device,
    #     verbose: bool = False,
    #     save_traj: bool = False,
    #     ddpm: bool = True,
    #     eta_ddim: float = 1.0,
    #     every_k_step: int = 1,
    #     ):
    #     pass
    
    
    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams["lr"],
                amsgrad=True,
                weight_decay=1.0e-12,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                nesterov=True,
            )
        if self.hparams["lr_scheduler"] == "reduce_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=self.hparams["lr_patience"],
                cooldown=self.hparams["lr_cooldown"],
                factor=self.hparams["lr_factor"],
            )
        elif self.hparams["lr_scheduler"] == "cyclic":
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.hparams["lr_min"],
                max_lr=self.hparams["lr"],
                mode="exp_range",
                step_size_up=self.hparams["lr_step_size"],
                cycle_momentum=False,
            )
        elif self.hparams["lr_scheduler"] == "one_cyclic":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams["lr"],
                steps_per_epoch=len(self.trainer.datamodule.train_dataset),
                epochs=self.hparams["num_epochs"],
            )
        elif self.hparams["lr_scheduler"] == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["lr_patience"],
                eta_min=self.hparams["lr_min"],
            )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": self.qed,
            "strict": False,
        }
        return [optimizer], [scheduler]

        