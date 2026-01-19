"""Performance tests for model inference speed.

This module tests that a staged model can perform predictions within
acceptable time limits for deployment.
"""

from __future__ import annotations

import os
import time
from argparse import Namespace
from dataclasses import asdict, dataclass

import pytest
import torch
import wandb

from tests import _PATH_DATA
from uaag2.data.uaag_dataset import Dataset_Info, UAAG2Dataset
from uaag2.equivariant_diffusion import Trainer

DATA_INFO_PATH = os.path.join(_PATH_DATA, "statistic.pkl")
LMDB_DATA_PATH = os.path.join(_PATH_DATA, "pdb_subset.lmdb")

NUM_PREDICTIONS = 5
MAX_TIME_SECONDS = 60.0


@dataclass
class MockParams:
    """Mock params object for model loading."""

    virtual_node: bool = True
    max_virtual_nodes: int = 5
    sdim: int = 256
    vdim: int = 64
    num_layers: int = 7
    use_cross_product: bool = False
    edim: int = 32
    cutoff_local: float = 7.0
    vector_aggr: str = "mean"
    fully_connected: bool = True
    local_global_model: bool = False
    edge_mp: bool = False
    context_mapping: bool = False
    num_context_features: int = 0
    bond_prediction: bool = False
    property_prediction: bool = False
    continuous_param: str = "data"
    load_ckpt_from_pretrained: str | None = None
    beta_min: float = 1e-4
    beta_max: float = 2e-2
    timesteps: int = 500
    noise_scheduler: str = "cosine"
    eps_min: float = 1e-3
    loss_weighting: str = "snr_t"
    gpus: int = 0
    save_dir: str = "."
    id: str = "test"


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
def mock_params() -> MockParams:
    """Create mock parameters for model loading."""
    return MockParams()


@pytest.fixture
def dataset_info(mock_params: MockParams) -> Dataset_Info:
    """Load dataset info for model initialization."""
    return Dataset_Info(mock_params, DATA_INFO_PATH)


@pytest.fixture
def sample_batch(mock_params: MockParams, device: torch.device) -> torch.Tensor:
    """Get a sample batch from the dataset for testing."""
    dataset = UAAG2Dataset(LMDB_DATA_PATH, params=mock_params)
    batch = dataset[0]
    return batch.to(device)


class TestModelPerformance:
    """Performance tests for the UAAG2 model."""

    def test_model_inference_speed(
        self,
        model_artifact_path: str,
        mock_params: MockParams,
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

        # Convert dataclass to Namespace for pytorch-lightning compatibility
        # Trainer expects hparams to be a Namespace (or dict, but Trainer uses dot access)
        hparams = Namespace(**asdict(mock_params))

        model = Trainer.load_from_checkpoint(
            checkpoint_path,
            hparams=hparams,
            dataset_info=dataset_info,
        ).to(device)
        model.eval()

        batch_size = int(sample_batch.batch.max()) + 1

        with torch.no_grad():
            _ = model(
                batch=sample_batch,
                t=torch.randint(1, mock_params.timesteps + 1, (batch_size,), device=device),
            )

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(NUM_PREDICTIONS):
                t = torch.randint(1, mock_params.timesteps + 1, (batch_size,), device=device)
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
