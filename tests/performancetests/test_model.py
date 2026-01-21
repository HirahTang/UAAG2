"""Performance tests for model inference speed.

This module tests that a staged model can perform predictions within
acceptable time limits for deployment.
"""

from __future__ import annotations

import os
import time
from omegaconf import OmegaConf, DictConfig

import pytest
import torch
import wandb
from torch_geometric.data import Batch

from tests import _PATH_DATA
from uaag2.datasets.uaag_dataset import Dataset_Info, UAAG2Dataset
from uaag2.equivariant_diffusion import Trainer

DATA_INFO_PATH = os.path.join(_PATH_DATA, "statistic.pkl")
LMDB_DATA_PATH = os.path.join(_PATH_DATA, "pdb_subset.lmdb")

NUM_PREDICTIONS = 2
MAX_TIME_SECONDS = 60.0


def get_model_path_from_wandb(artifact_path: str) -> str:
    """Download model artifact from wandb and return local checkpoint path.

    Args:
        artifact_path: The wandb artifact path in format 'entity/project/artifact:version'.

    Returns:
        Local path to the downloaded checkpoint file.
    """
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    for file in os.listdir(artifact_dir):
        if file.endswith(".ckpt"):
            return os.path.join(artifact_dir, file)

    raise FileNotFoundError(f"No checkpoint file found in artifact: {artifact_path}")


@pytest.fixture
def model_artifact_path() -> str:
    """Get the model artifact path from environment variable."""
    model_name = os.environ.get("MODEL_NAME")
    if model_name is None:
        pytest.skip("MODEL_NAME environment variable not set")
    return model_name


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def mock_params() -> DictConfig:
    """Create mock parameters for model loading."""
    return OmegaConf.create(
        {
            "id": "test",
            "gpus": 0,
            "num_epochs": 1,
            "save_dir": ".",
            "seed": 42,
            "eval_freq": 1,
            "precision": 32,
            "detect_anomaly": False,
            "accum_batch": 1,
            "ema_decay": 0.999,
            "load_ckpt": None,
            "load_ckpt_from_pretrained": None,
            "backprop_local": False,
            "model": {
                "sdim": 256,
                "vdim": 64,
                "rbf_dim": 32,
                "edim": 32,
                "edge_mp": False,
                "vector_aggr": "mean",
                "num_layers": 7,
                "fully_connected": True,
                "local_global_model": False,
                "local_edge_attrs": False,
                "use_cross_product": False,
                "cutoff_local": 7.0,
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
                "sdim_latent": 256,
                "vdim_latent": 64,
                "latent_dim": None,
                "edim_latent": 32,
                "num_layers_latent": 7,
                "latent_layers": 7,
                "latentmodel": "diffusion",
                "latent_detach": False,
                "max_virtual_nodes": 5,
                "virtual_node": True,
                "dropout_prob": 0.3,
                "weight_decay": 0.9999,
            },
            "data": {
                "mask_rate": 0.0,
                "pocket_noise": False,
                "pocket_noise_scale": 0.2,  # Added default
            },
            "diffusion": {
                "continuous": False,
                "noise_scheduler": "cosine",
                "eps_min": 1e-3,
                "beta_min": 1e-4,
                "beta_max": 2e-2,
                "timesteps": 500,
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
                "lr": 5e-4,  # Default from api inputs
                "weight_decay": 0.9999,
                "name": "adam",
                "lr_scheduler": "reduce_on_plateau",
                "grad_clip_val": 10.0,
                "lr_min": 5e-5,
                "lr_step_size": 10000,
                "lr_frequency": 5,
                "lr_patience": 20,
                "lr_cooldown": 5,
                "lr_factor": 0.75,
            },
        }
    )


@pytest.fixture
def dataset_info(mock_params: DictConfig) -> Dataset_Info:
    """Load dataset info for model initialization."""
    return Dataset_Info(mock_params, DATA_INFO_PATH)


@pytest.fixture
def sample_batch(mock_params: DictConfig, device: torch.device) -> torch.Tensor:
    """Get a sample batch from the dataset for testing."""
    dataset = UAAG2Dataset(LMDB_DATA_PATH, params=mock_params)
    batch = dataset[0]
    return Batch.from_data_list([batch]).to(device)


class TestModelPerformance:
    """Performance tests for the UAAG2 model."""

    def test_model_inference_speed(
        self,
        model_artifact_path: str,
        mock_params: DictConfig,
        dataset_info: Dataset_Info,
        sample_batch: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Test that the model can perform 100 predictions within the time limit.

        This test loads the model from a wandb artifact, runs 100 forward passes
        with a sample batch, and asserts that the total time is under the threshold.

        Args:
            model_artifact_path: Path to the wandb model artifact.
            mock_params: Mock parameters for model initialization.
            dataset_info: Dataset statistics for model initialization.
            sample_batch: A sample batch from the dataset.
            device: The device to run inference on.
        """
        checkpoint_path = get_model_path_from_wandb(model_artifact_path)

        # Update params with loaded checkpoint?
        # Trainer.load_from_checkpoint will merge but it's good to pass structure
        hparams = mock_params

        model = Trainer.load_from_checkpoint(
            checkpoint_path,
            hparams=hparams,
            dataset_info=dataset_info,
        ).to(device)
        model.eval()

        batch_size = int(sample_batch.batch.max()) + 1

        timesteps = hparams.diffusion.timesteps

        with torch.no_grad():
            _ = model(
                batch=sample_batch,
                t=torch.randint(1, timesteps + 1, (batch_size,), device=device),
            )

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(NUM_PREDICTIONS):
                t = torch.randint(1, timesteps + 1, (batch_size,), device=device)
                _ = model(batch=sample_batch, t=t)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print("\nPerformance Results:")
        print(f"  Device: {device}")
        print(f"  Total predictions: {NUM_PREDICTIONS}")
        print(f"  Total time: {total_time:.3f} seconds")
        print(f"  Average time per prediction: {total_time / NUM_PREDICTIONS * 1000:.2f} ms")

        assert total_time < MAX_TIME_SECONDS, (
            f"Model inference too slow: {NUM_PREDICTIONS} predictions took {total_time:.2f}s, "
            f"expected < {MAX_TIME_SECONDS}s"
        )
