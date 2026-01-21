import torch
from argparse import Namespace
from uaag2.model import Trainer


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
    hparams = Namespace(
        sdim=32,
        vdim=8,
        rbf_dim=8,
        edim=8,
        edge_mp=False,
        vector_aggr="mean",
        num_layers=2,
        fully_connected=True,
        local_global_model=False,
        local_edge_attrs=False,
        use_cross_product=False,
        cutoff_local=5.0,
        cutoff_global=10.0,
        energy_training=False,
        property_training=False,
        regression_property="polarizability",
        energy_loss="l2",
        use_pos_norm=False,
        additional_feats=True,
        use_qm_props=False,
        build_mol_with_addfeats=False,
        continuous=False,
        noise_scheduler="cosine",
        eps_min=1e-3,
        beta_min=1e-4,
        beta_max=2e-2,
        timesteps=10,
        max_time=None,
        lc_coords=3.0,
        lc_atoms=0.4,
        lc_bonds=2.0,
        lc_charges=1.0,
        lc_mulliken=1.5,
        lc_wbo=2.0,
        loss_weighting="snr_t",
        snr_clamp_min=0.05,
        snr_clamp_max=1.50,
        ligand_pocket_interaction=False,
        diffusion_pretraining=False,
        continuous_param="data",
        atoms_categorical=True,
        bonds_categorical=True,
        atom_type_masking=True,
        use_absorbing_state=False,
        num_bond_classes=5,
        num_charge_classes=6,
        pocket_noise=False,
        mask_rate=0.5,
        pocket_noise_scale=0.01,
        bond_guidance_model=False,
        bond_prediction=False,
        bond_model_guidance=False,
        energy_model_guidance=False,
        polarizabilty_model_guidance=False,
        ckpt_bond_model=None,
        ckpt_energy_model=None,
        ckpt_polarizabilty_model=None,
        guidance_scale=1.0e-4,
        context_mapping=False,
        num_context_features=0,
        properties_list=[],
        property_prediction=False,
        prior_beta=1.0,
        sdim_latent=32,
        vdim_latent=8,
        latent_dim=None,
        edim_latent=8,
        num_layers_latent=2,
        latent_layers=2,
        latentmodel="diffusion",
        latent_detach=False,
        lr=1e-3,
        batch_size=4,
        num_epochs=1,
        learning_rate=1e-3,  # Added for wandb logging compatibility inside Trainer if needed, though mostly used in train.py
        seed=42,
        backprop_local=False,  # Added as it was present in train.py default args
        ema_decay=0.999,
        weight_decay=0.0,
        gpus=0,
        accum_batch=1,
        eval_freq=1,
        grad_clip_val=1.0,
        precision=32,
        detect_anomaly=False,
        n_draws=10,  # Just in case it's used
        save_dir="/tmp/test_uaag2_logs",  # Added save_dir which was missing
        id="test_run",  # Added id which was missing
        load_ckpt=None,  # Added load_ckpt
        load_ckpt_from_pretrained=None,  # Added load_ckpt_from_pretrained
    )

    dataset_info = MockDatasetInfo()

    model = Trainer(hparams=hparams, dataset_info=dataset_info)

    assert model is not None
    print("Model constructed successfully")
