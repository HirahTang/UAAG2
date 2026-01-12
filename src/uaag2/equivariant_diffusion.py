import json
import logging
import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from tqdm import tqdm

from uaag2.diffusion.categorical import CategoricalDiffusionKernel
from uaag2.diffusion.continuous import DiscreteDDPM
from uaag2.e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from uaag2.losses import DiffusionLoss
from uaag2.utils import (
    convert_edge_to_bond,
    get_molecules,
    initialize_edge_attrs_reverse,
    load_model,
    write_xyz_file_from_batch,
)


logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(logging.NullHandler())


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
        self.validity = 0.0

        self.save_dir = os.path.join(hparams.save_dir, f"run{hparams.id}")
        self.qed = 0.0
        self.dataset_info = dataset_info
        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.bond_types.float()
        charge_types_distribution = dataset_info.charge_types.float()
        is_aromatic_distribution = dataset_info.is_aromatic.float()
        is_ring_distribution = dataset_info.is_ring.float()
        hybridization_distribution = dataset_info.hybridization.float()
        degree_distribution = dataset_info.degree.float()

        # Implementation of the absorbing states
        atom_types_distribution = torch.zeros_like(atom_types_distribution)
        atom_types_distribution[-1] = 1.0

        bond_types_distribution = torch.zeros_like(bond_types_distribution)
        bond_types_distribution[0] = 1.0

        # charge_types_distribution = torch.zeros_like(charge_types_distribution)
        # charge_types_distribution[-1] = 1.0

        # is_aromatic_distribution = torch.zeros_like(is_aromatic_distribution)
        # is_aromatic_distribution[-1] = 1.0

        # is_ring_distribution = torch.zeros_like(is_ring_distribution)
        # is_ring_distribution[-1] = 1.0

        # hybridization_distribution = torch.zeros_like(hybridization_distribution)
        # hybridization_distribution[-1] = 1.0

        # degree_distribution = torch.zeros_like(degree_distribution)
        # degree_distribution[-1] = 1.0

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())
        self.register_buffer("is_aromatic_prior", is_aromatic_distribution.clone())
        self.register_buffer("is_in_ring_prior", is_ring_distribution.clone())
        self.register_buffer("hybridization_prior", hybridization_distribution.clone())
        self.register_buffer("degree_prior", degree_distribution.clone())

        self.num_is_aromatic = len(is_aromatic_distribution)
        self.num_is_in_ring = len(is_ring_distribution)
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
                self.num_atom_features,
                self.num_bond_classes,
            )
            # num_params = len(self.model.state_dict())
            # for i, param in enumerate(self.model.parameters()):
            #     if i < num_params // 2:
            #         param.requires_grad = False
        else:
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
            modalities=[
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
        # print("Defined Trainer")
        # from IPython import embed; embed()

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        try:
            loss = self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            loss = torch.tensor(0.0, device=batch.x.device, requires_grad=True)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
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
            weights = self.sde_bonds.snr_s_t_weighting(s=t - 1, t=t, clamp_min=None, clamp_max=None).to(batch.x.device)
        elif self.hparams.loss_weighting == "snr_t":
            weights = self.sde_bonds.snr_t_weighting(t=t, device=batch.x.device, clamp_min=0.05, clamp_max=5.0)
        elif self.hparams.loss_weighting == "exp_t":
            weights = self.sde_atom_charge.exp_t_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "exp_t_half":
            weights = self.sde_atom_charge.exp_t_half_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "uniform":
            weights = None

        ligand_mask = (batch.is_ligand - batch.is_backbone).long()

        # skip the current batch if batch.x.shape[0] > 1600
        # if batch.x.shape[0] > 1600:
        #     print(f"Skipping batch {batch_idx} with atom size {batch.x.shape[0]}")
        #     return torch.tensor(0.0, device=batch.x.device, requires_grad=True)

        # if batch.edge_index.shape[1] > 270000:
        #     print(f"Skipping batch {batch_idx} with edge size {batch.edge_index.shape[1]}")
        #     return torch.tensor(0.0, device=batch.x.device, requires_grad=True)

        out_dict = self(batch=batch, t=t)

        if out_dict is None:
            return torch.tensor(0.0, device=batch.x.device, requires_grad=True)

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

        (atoms_pred, charges_pred, ring_pred, aromatic_pred, hybridization_pred, degree_pred, _) = atoms_pred.split(
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

        loss = self.diffusion_loss(
            pred_data=pred_data,
            true_data=true_data,
            batch=batch.batch[ligand_mask == 1],
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
            loss["degree"],
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
        noise_coords_true = noise_coords_true[ligand_mask == 1]
        pos_true = pos[ligand_mask == 1]

        atom_types, atom_types_perturbed = self.cat_atoms.sample_categorical(
            t,
            atom_types,
            data_batch,
            self.dataset_info,
            num_classes=self.num_atom_types,
            type="atoms",
        )

        atom_types_perturbed = atom_types_perturbed * ligand_mask.unsqueeze(1) + atom_types * pocket_mask.unsqueeze(1)
        atom_types = atom_types[ligand_mask == 1]
        charges, charges_perturbed = self.cat_charges.sample_categorical(
            t,
            charges,
            data_batch,
            self.dataset_info,
            num_classes=self.num_charge_classes,
            type="charges",
        )

        charges_perturbed = charges_perturbed * ligand_mask.unsqueeze(1) + charges * pocket_mask.unsqueeze(1)
        charges = charges[ligand_mask == 1]
        edge_attr_global_perturbed, edge_attr_global_original = self.cat_bonds.sample_edges_categorical(
            t, bond_edge_index, bond_edge_attr, data_batch
        )

        edge_attr_global_perturbed = edge_attr_global_perturbed.float()
        edge_attr_global_original = edge_attr_global_original.float()
        edge_attr_global_perturbed = edge_attr_global_perturbed * batch.edge_ligand.unsqueeze(
            1
        ) + edge_attr_global_original * (1 - batch.edge_ligand).unsqueeze(1)
        edge_attr_global_original = edge_attr_global_original[batch.edge_ligand == 1]

        ring_feat, ring_feat_perturbed = self.cat_ring.sample_categorical(
            t,
            ring_feat,
            data_batch,
            self.dataset_info,
            num_classes=self.num_is_in_ring,
            type="ring",
        )
        ring_feat_perturbed = ring_feat_perturbed * ligand_mask.unsqueeze(1) + ring_feat * pocket_mask.unsqueeze(1)
        ring_feat = ring_feat[ligand_mask == 1]

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
        aromatic_feat_perturbed = aromatic_feat_perturbed * ligand_mask.unsqueeze(
            1
        ) + aromatic_feat * pocket_mask.unsqueeze(1)
        aromatic_feat = aromatic_feat[ligand_mask == 1]

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
        hybridization_feat_perturbed = hybridization_feat_perturbed * ligand_mask.unsqueeze(
            1
        ) + hybridization_feat * pocket_mask.unsqueeze(1)
        hybridization_feat = hybridization_feat[ligand_mask == 1]

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
        degree_feat_perturbed = degree_feat_perturbed * ligand_mask.unsqueeze(1) + degree_feat * pocket_mask.unsqueeze(
            1
        )
        degree_feat = degree_feat[ligand_mask == 1]

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

        try:
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
                batch_lig=batch.batch[pocket_mask == 0],
            )
        except torch.cuda.OutOfMemoryError:
            # Compact log instead of full traceback:

            print(
                f"[OOM {datetime.now()}]"
                f"nodes_total={atom_feats_in_perturbed.shape[0]} "
                f"edges_total={edge_attr_global_perturbed.shape[0]}",
                file=sys.stdout,
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Skip this batch and keep training:
            return None

        out["coords_perturbed"] = pos_perturbed[ligand_mask == 1]
        out["atoms_perturbed"] = atom_types_perturbed[ligand_mask == 1]
        out["charges_perturbed"] = charges_perturbed[ligand_mask == 1]
        out["bonds_perturbed"] = edge_attr_global_perturbed[batch.edge_ligand == 1]
        out["ring_perturbed"] = ring_feat_perturbed[ligand_mask == 1]
        out["aromatic_perturbed"] = aromatic_feat_perturbed[ligand_mask == 1]
        out["hybridization_perturbed"] = hybridization_feat_perturbed[ligand_mask == 1]
        out["degree_perturbed"] = degree_feat_perturbed[ligand_mask == 1]

        out["coords_true"] = pos_true
        out["coords_noise_true"] = noise_coords_true
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global_original
        out["charges_true"] = charges.argmax(dim=-1)
        out["ring_true"] = ring_feat.argmax(dim=-1)
        out["aromatic_true"] = aromatic_feat.argmax(dim=-1)
        out["hybridization_true"] = hybridization_feat.argmax(dim=-1)
        out["degree_true"] = degree_feat.argmax(dim=-1)
        out["bond_aggregation_index"] = bond_edge_index[1][batch.edge_ligand == 1]
        out["edge_batch"] = batch.batch[bond_edge_index[0]][batch.edge_ligand == 1]

        return out

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:
            if self.local_rank == 0:
                print(f"Running evaluation in epoch {self.current_epoch + 1}")
            final_res = self.run_evaluation(
                step=self.i,
                device="cuda" if self.hparams.gpus > 1 else "cpu",
                dataset_info=self.dataset_info,
                ngraphs=64,
                bs=self.hparams.inference_batch_size,
                verbose=True,
                inner_verbose=False,
                eta_ddim=1.0,
                ddpm=True,
                every_k_step=1,
            )

            self.i += 1
            self.log(
                "validity",
                final_res["validity"],
                on_epoch=True,
                sync_dist=True,
            )

            self.log(
                "connectivity",
                final_res["connectivity"],
                on_epoch=True,
                sync_dist=True,
            )

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

    @torch.no_grad()
    def run_evaluation(
        self,
        step: int,
        dataset_info,
        ngraphs: int = 4000,
        bs: int = 500,
        return_molecules: bool = False,
        verbose: bool = True,
        inner_verbose=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        run_test_eval: bool = False,
        save_traj: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        dataloader = self.trainer.datamodule.test_dataloader()

        molecule_list = []
        connected_list = []
        sanitized_list = []
        start = datetime.now()

        for i, batch in enumerate(dataloader):
            molecules, connected_ele, sanitized_ele = self.reverse_sampling(
                batch=batch,
                device=device,
                verbose=verbose,
                save_traj=save_traj,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
                iteration=i,
                show_pocket=True,
            )

            molecule_list.extend(molecules)
            connected_list.extend(connected_ele)
            sanitized_list.extend(sanitized_ele)
        connect_rate = torch.tensor(connected_list).sum().item() / len(connected_list)
        sanitize_rate = torch.tensor(sanitized_list).sum().item() / len(sanitized_list)

        if not run_test_eval:
            save_cond = self.validity < sanitize_rate and self.connected_components < connect_rate
        else:
            save_cond = False

        if save_cond:
            self.validity = sanitize_rate
            self.connected_components = connect_rate
            save_path = os.path.join(self.hparams.save_dir, "best_valid.ckpt")
            self.trainer.save_checkpoint(save_path)

        run_time = datetime.now() - start

        if verbose:
            if self.local_rank == 0:
                print(f"Run time={run_time}")

        total_res = {"validity": sanitize_rate, "connectivity": connect_rate}
        if self.local_rank == 0:
            print(total_res)

        total_res["step"] = str(step)
        total_res["epoch"] = str(self.current_epoch)
        total_res["run_time"] = str(run_time)

        if return_molecules:
            return molecule_list, total_res
        else:
            return total_res

    @torch.no_grad()
    def generate_ligand(
        self,
        loader,
        save_path: str = "somename",
        verbose: bool = False,
        save_traj: bool = False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        show_pocket: bool = False,
        device: str = "cpu",
    ):
        print("Number of graphs: ", len(loader))
        molecule_list = []
        connected_list = []
        sanitized_list = []
        start = datetime.now()

        for i, batch in enumerate(loader):
            reconstruct_mask = batch.is_ligand - batch.is_backbone
            molecules = self.reverse_sampling(
                batch=batch,
                device=device,
                verbose=verbose,
                save_traj=save_traj,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
                iteration=i,
                show_pocket=False,
            )
            connected_list_batch, sanitized_list_batch = convert_edge_to_bond(
                batch=batch,
                out_dict=molecules,
                path=os.path.join(self.save_dir, save_path, f"iter_{i}"),
                reconstruct_mask=reconstruct_mask,
                atom_decoder=self.dataset_info.atom_decoder,
                edge_decoder=self.dataset_info.bond_decoder,
            )

            molecule_list.extend(molecules)
            connected_list.extend(connected_list_batch)
            sanitized_list.extend(sanitized_list_batch)

        connect_rate = torch.tensor(connected_list).sum().item() / len(connected_list)
        sanitize_rate = torch.tensor(sanitized_list).sum().item() / len(sanitized_list)

        run_time = datetime.now() - start

        if verbose:
            print(f"Run time={run_time}")

        # save connect_rate and sanitize_rate to a json file
        total_res = {"validity": sanitize_rate, "connectivity": connect_rate}
        print(total_res)

        with open(os.path.join(self.save_dir, save_path, "results.json"), "w") as f:
            json.dump(total_res, f)

    def reverse_sampling(
        self,
        batch,
        device: torch.device,
        verbose: bool = False,
        save_traj: bool = False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        iteration: int = 0,
        show_pocket: bool = False,
    ):
        # implement empirical_distribution_num_nodes of ligand node (randomly initiated)
        # back to the graph, with fully connected edges
        batch = batch.to(self.device)
        reconstruct_mask = batch.is_ligand - batch.is_backbone

        pos_ligand = batch.pos[reconstruct_mask == 1]

        pos = torch.randn_like(pos_ligand)
        n = pos_ligand.size(0)

        compound_pos = batch.pos
        compound_pos[reconstruct_mask == 1] = pos
        compound_pos = compound_pos.to(self.device)

        atom_types = torch.multinomial(self.atoms_prior, num_samples=n, replacement=True).to(self.device)
        atom_types_ligand = batch.x[reconstruct_mask == 1]

        compound_atom_types = batch.x.long().to(self.device)
        compound_atom_types[reconstruct_mask == 1] = atom_types
        compound_atom_types = F.one_hot(compound_atom_types, num_classes=self.num_atom_types).float()
        atom_types = F.one_hot(atom_types, num_classes=self.num_atom_types).float()

        charge_types = torch.multinomial(self.charges_prior, num_samples=n, replacement=True).to(self.device)
        compound_charges = batch.charges.long().to(self.device)
        compound_charges[reconstruct_mask == 1] = charge_types
        compound_charges = F.one_hot(compound_charges, num_classes=self.num_charge_classes).float()
        charge_types = F.one_hot(charge_types, num_classes=self.num_charge_classes).float()

        # ring
        ring_feat = torch.multinomial(self.is_in_ring_prior, num_samples=n, replacement=True).to(self.device)

        compound_ring_feat = batch.is_in_ring.long().to(self.device)
        compound_ring_feat[reconstruct_mask == 1] = ring_feat
        compound_ring_feat = F.one_hot(compound_ring_feat, num_classes=self.num_is_in_ring).float()
        ring_feat = F.one_hot(ring_feat, num_classes=self.num_is_in_ring).float()

        # aromatic
        aromatic_feat = torch.multinomial(self.is_aromatic_prior, num_samples=n, replacement=True).to(self.device)

        compound_aromatic_feat = batch.is_aromatic.long().to(self.device)
        compound_aromatic_feat[reconstruct_mask == 1] = aromatic_feat
        compound_aromatic_feat = F.one_hot(compound_aromatic_feat, num_classes=self.num_is_aromatic).float()
        aromatic_feat = F.one_hot(aromatic_feat, num_classes=self.num_is_aromatic).float()

        # hybridization
        hybridization_feat = torch.multinomial(self.hybridization_prior, num_samples=n, replacement=True).to(
            self.device
        )

        compound_hybridization_feat = batch.hybridization.long().to(self.device)
        compound_hybridization_feat[reconstruct_mask == 1] = hybridization_feat
        compound_hybridization_feat = F.one_hot(compound_hybridization_feat, num_classes=self.num_hybridization).float()
        hybridization_feat = F.one_hot(hybridization_feat, num_classes=self.num_hybridization).float()

        # degree
        degree_feat = torch.multinomial(self.degree_prior, num_samples=n, replacement=True).to(self.device)

        compound_degree_feat = batch.degree.long().to(self.device)
        compound_degree_feat[reconstruct_mask == 1] = degree_feat
        compound_degree_feat = F.one_hot(compound_degree_feat, num_classes=self.num_degree).float()
        degree_feat = F.one_hot(degree_feat, num_classes=self.num_degree).float()

        edge_index_ligand = batch.edge_index.t()[batch.edge_ligand == 1].t()

        (
            edge_attr_global_lig,
            edge_index_global_lig,
            mask,
            mask_i,
        ) = initialize_edge_attrs_reverse(
            edge_index_ligand,
            n,
            self.bonds_prior,
            self.num_bond_classes,
            self.device,
        )

        edge_attr_full = batch.edge_attr.long().to(self.device)
        edge_attr_full = F.one_hot(edge_attr_full, num_classes=self.num_bond_classes).float()

        edge_attr_full[batch.edge_ligand == 1] = edge_attr_global_lig

        edge_index_global = batch.edge_index.to(self.device)

        batch_is_ligand = batch.is_ligand.unsqueeze(1).to(self.device)
        atoms_feats_in_perturbed = torch.cat(
            [
                compound_atom_types,
                compound_charges,
                compound_ring_feat,
                compound_aromatic_feat,
                compound_hybridization_feat,
                compound_degree_feat,
                batch_is_ligand,
            ],
            dim=-1,
        )

        pocket_mask = (1 - batch.is_ligand + batch.is_backbone).long()

        if self.hparams.continuous_param == "data":
            chain = range(0, self.hparams.timesteps)
        elif self.hparams.continuous_param == "noise":
            chain = range(0, self.hparams.timesteps - 1)

        chain = chain[::every_k_step]

        iterator = tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        batch_lig = batch.batch[pocket_mask == 0]
        for i, timestep in enumerate(iterator):
            s = torch.full(size=(n,), fill_value=timestep, dtype=torch.long, device=compound_pos.device)
            t = s + 1

            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)

            out = self.model(
                x=atoms_feats_in_perturbed,
                t=temb,
                pos=compound_pos,
                edge_index_local=None,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_full,
                batch=batch.batch,
                batch_edge_global=batch.edge_ligand.long(),
                context=None,
                pocket_mask=pocket_mask.unsqueeze(1),
                edge_mask=batch.edge_ligand.long(),
                batch_lig=batch.batch[pocket_mask == 0],
            )

            coords_pred = out["coords_pred"].squeeze()

            (atoms_pred, charges_pred, ring_pred, aromatic_pred, hybridization_pred, degree_pred, _) = out[
                "atoms_pred"
            ].split(
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

            atoms_pred = atoms_pred.softmax(dim=-1)

            edges_pred = out["bonds_pred"].softmax(dim=-1)

            charges_pred = charges_pred.softmax(dim=-1)
            ring_pred = ring_pred.softmax(dim=-1)
            aromatic_pred = aromatic_pred.softmax(dim=-1)
            hybridization_pred = hybridization_pred.softmax(dim=-1)
            degree_pred = degree_pred.softmax(dim=-1)

            if ddpm:
                if self.hparams.noise_scheduler == "adaptive":
                    # positions
                    pos = self.sde_pos.sample_reverse_adaptive(
                        s, t, pos, coords_pred, batch_lig, cog_proj=False, eta_ddim=eta_ddim
                    )  # here is cog_proj false as it will be downprojected later
                else:
                    # positions
                    pos = self.sde_pos.sample_reverse(
                        t, pos, coords_pred, batch_lig, cog_proj=False, eta_ddim=eta_ddim
                    )  # here is cog_proj false as it will be downprojected later
            else:
                pos = self.sde_pos.sample_reverse_ddim(
                    t, pos, coords_pred, batch_lig, cog_proj=False, eta_ddim=eta_ddim
                )  # here is cog_proj false as it will be downprojected later

            # atoms

            atom_types = self.cat_atoms.sample_reverse_categorical(
                xt=atom_types,
                x0=atoms_pred,
                t=t[batch_lig],
                num_classes=self.num_atom_types,
            )

            # charges
            charge_types = self.cat_charges.sample_reverse_categorical(
                xt=charge_types,
                x0=charges_pred,
                t=t[batch_lig],
                num_classes=self.num_charge_classes,
            )

            # additional feats
            ring_feat = self.cat_ring.sample_reverse_categorical(
                xt=ring_feat,
                x0=ring_pred,
                t=t[batch_lig],
                num_classes=self.num_is_in_ring,
            )

            aromatic_feat = self.cat_aromatic.sample_reverse_categorical(
                xt=aromatic_feat,
                x0=aromatic_pred,
                t=t[batch_lig],
                num_classes=self.num_is_aromatic,
            )
            hybridization_feat = self.cat_hybridization.sample_reverse_categorical(
                xt=hybridization_feat,
                x0=hybridization_pred,
                t=t[batch_lig],
                num_classes=self.num_hybridization,
            )

            degree_feat = self.cat_degree.sample_reverse_categorical(
                xt=degree_feat,
                x0=degree_pred,
                t=t[batch_lig],
                num_classes=self.num_degree,
            )

            (
                edge_attr_global_lig,
                edge_index_global_lig,
                mask,
                mask_i,
            ) = self.cat_bonds.sample_reverse_edges_categorical(
                edge_attr_global_lig,
                edges_pred,
                t,
                mask,
                mask_i,
                batch=batch.batch[edge_index_ligand[0]],
                edge_index_global=edge_index_global_lig,
                num_classes=self.num_bond_classes,
            )

            # combine the denoised features with the pocket features

            compound_atom_types[reconstruct_mask == 1] = atom_types
            compound_charges[reconstruct_mask == 1] = charge_types
            compound_ring_feat[reconstruct_mask == 1] = ring_feat
            compound_aromatic_feat[reconstruct_mask == 1] = aromatic_feat
            compound_hybridization_feat[reconstruct_mask == 1] = hybridization_feat
            compound_degree_feat[reconstruct_mask == 1] = degree_feat

            compound_pos[reconstruct_mask == 1] = pos

            edge_attr_full[batch.edge_ligand == 1] = edge_attr_global_lig

            atoms_feats_in_perturbed = torch.cat(
                [
                    compound_atom_types,
                    compound_charges,
                    compound_ring_feat,
                    compound_aromatic_feat,
                    compound_hybridization_feat,
                    compound_degree_feat,
                    batch_is_ligand,
                ],
                dim=-1,
            )
            if save_traj:
                atom_decoder = self.dataset_info.atom_decoder
                write_xyz_file_from_batch(
                    pos=compound_pos,
                    atom_type=compound_atom_types,
                    batch=batch.batch,
                    atom_decoder=atom_decoder,
                    path=os.path.join(self.save_dir, f"epoch_{self.current_epoch}", f"iter_{iteration}"),
                    i=i,
                )

        out_dict = {
            "coords_pred": pos,
            "atoms_pred": atom_types,
            "charges_pred": charge_types,
            "bonds_pred": edge_attr_global_lig,
            "coords_pocket": compound_pos[batch.is_ligand == 0],
            "atoms_pocket": compound_atom_types[batch.is_ligand == 0],
            "aromatic_pred": aromatic_feat,
            "hybridization_pred": hybridization_feat,
            "coords_backbone": compound_pos[batch.is_backbone == 1],
            "atoms_backbone": compound_atom_types[batch.is_backbone == 1],
            "coords_true": pos_ligand,
            "atoms_true": atom_types_ligand,
        }

        if show_pocket:
            connected_list, sanitized_list = convert_edge_to_bond(
                batch=batch,
                out_dict=out_dict,
                path=os.path.join(self.save_dir, f"epoch_{self.current_epoch}", f"iter_{iteration}"),
                reconstruct_mask=reconstruct_mask,
                atom_decoder=self.dataset_info.atom_decoder,
                edge_decoder=self.dataset_info.bond_decoder,
            )
            try:
                _, _ = get_molecules(
                    out_dict=out_dict,
                    path=os.path.join(self.save_dir, f"epoch_{self.current_epoch}", f"iter_{iteration}"),
                    batch=batch.batch,
                    reconstruct_mask=reconstruct_mask,
                    backbone_mask=batch.is_backbone,
                    pocket_mask=batch.is_ligand,
                    atom_decoder=self.dataset_info.atom_decoder,
                )
            except Exception:
                print("No Pocket for this molecule")
                return out_dict, connected_list, sanitized_list
            return out_dict, connected_list, sanitized_list
        else:
            return out_dict
            # create the input to the network of the next timestep

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
