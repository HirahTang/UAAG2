import torch
from omegaconf import OmegaConf
from uaag2.equivariant_diffusion import Trainer


class MockDatasetInfo:
    def __init__(self):
        self.atom_types = torch.tensor([0.5, 0.5])
        self.bond_types = torch.tensor([0.25, 0.25, 0.25, 0.25])
        self.charge_types = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        self.is_aromatic = torch.tensor([0.5, 0.5])
        self.is_ring = torch.tensor([0.5, 0.5])
        self.hybridization = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        self.degree = torch.tensor([0.1] * 10)
        self.atom_decoder = {0: "C", 1: "N"}
        self.bond_decoder = {1: "SINGLE", 2: "DOUBLE"}


def test_model_construction():
    hparams = OmegaConf.create(
        {
            "id": "test_run",
            "gpus": 0,
            "num_epochs": 1,
            "save_dir": "/tmp/test_uaag2_logs",
            "seed": 42,
            "eval_freq": 1,
            "precision": 32,
            "detect_anomaly": False,
            "accum_batch": 1,
            "ema_decay": 0.999,
            "load_ckpt": None,
            "load_ckpt_from_pretrained": None,
            "model": {
                "sdim": 32,
                "vdim": 8,
                "rbf_dim": 8,
                "edim": 8,
                "edge_mp": False,
                "vector_aggr": "mean",
                "num_layers": 2,
                "fully_connected": True,
                "local_global_model": False,
                "local_edge_attrs": False,
                "use_cross_product": False,
                "cutoff_local": 5.0,
                "cutoff_global": 10.0,
                "energy_training": False,
                "property_training": False,
                "regression_property": "polarizability",
                "energy_loss": "l2",
                "use_pos_norm": False,
                "additional_feats": True,
                "use_qm_props": False,
                "build_mol_with_addfeats": False,
                "bond_guidance_model": False,
                "bond_prediction": False,
                "bond_model_guidance": False,
                "energy_model_guidance": False,
                "polarizabilty_model_guidance": False,
                "ckpt_bond_model": None,
                "ckpt_energy_model": None,
                "ckpt_polarizabilty_model": None,
                "guidance_scale": 1.0e-4,
                "context_mapping": False,
                "num_context_features": 0,
                "properties_list": [],
                "property_prediction": False,
                "prior_beta": 1.0,
                "sdim_latent": 32,
                "vdim_latent": 8,
                "latent_dim": None,
                "edim_latent": 8,
                "num_layers_latent": 2,
                "latent_layers": 2,
                "latentmodel": "diffusion",
                "latent_detach": False,
                "backprop_local": False,
                "dropout_prob": 0.0,  # Added default
                "weight_decay": 0.0,
                "virtual_node": False,
                "max_virtual_nodes": 5,
            },
            "data": {
                "batch_size": 4,
                "mask_rate": 0.5,
                "pocket_noise": False,
                "pocket_noise_scale": 0.01,
                "backprop_local": False,  # Double check where this belongs, seemingly model or data, but usually model
            },
            "diffusion": {
                "continuous": False,
                "noise_scheduler": "cosine",
                "eps_min": 1e-3,
                "beta_min": 1e-4,
                "beta_max": 2e-2,
                "timesteps": 10,
                "max_time": None,
                "lc_coords": 3.0,
                "lc_atoms": 0.4,
                "lc_bonds": 2.0,
                "lc_charges": 1.0,
                "lc_mulliken": 1.5,
                "lc_wbo": 2.0,
                "loss_weighting": "snr_t",
                "snr_clamp_min": 0.05,
                "snr_clamp_max": 1.50,
                "ligand_pocket_interaction": False,
                "diffusion_pretraining": False,
                "continuous_param": "data",
                "atoms_categorical": True,
                "bonds_categorical": True,
                "atom_type_masking": True,
                "use_absorbing_state": False,
                "num_bond_classes": 5,
                "num_charge_classes": 6,
            },
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "name": "adam",  # Added default
                "lr_scheduler": "reduce_on_plateau",  # Added default
                "lr_patience": 20,
                "lr_cooldown": 5,
                "lr_factor": 0.75,
                "lr_frequency": 5,
                "grad_clip_val": 1.0,
            },
        }
    )

    dataset_info = MockDatasetInfo()

    model = Trainer(hparams=hparams, dataset_info=dataset_info)

    assert model is not None
    print("Model constructed successfully")
