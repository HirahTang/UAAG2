import os
from dataclasses import dataclass

import lmdb
import pytest
from torch.utils.data import Dataset

param = pytest.mark.parametrize

from uaag2.data.uaag_dataset import UAAG2Dataset

from tests import _PATH_DATA


DATA_PATH = os.path.join(_PATH_DATA, "pdb_subset.lmdb")


@dataclass
class MockParams:
    """Mock params object for testing."""

    virtual_node: bool = False


class TestUAAG2Dataset:
    """Basic tests for the UAAG2Dataset class."""

    def test_dataset_instantiation(self):
        """Test that the dataset can be instantiated."""
        dataset = UAAG2Dataset(DATA_PATH)
        assert isinstance(dataset, Dataset)

    def test_dataset_length(self):
        """Test that __len__ returns a positive integer."""
        dataset = UAAG2Dataset(DATA_PATH)
        length = len(dataset)
        assert isinstance(length, int)
        assert length > 0

    def test_dataset_getitem(self):
        """Test that we can retrieve an item from the dataset."""
        dataset = UAAG2Dataset(DATA_PATH, params=MockParams())
        item = dataset[0]
        assert item is not None

    @param("mask_rate", [0.0, 0.5, 1.0])
    @param("noise_scale", [0.1, 0.5])
    def test_dataset_params(self, mask_rate, noise_scale):
        """Test that mask_rate and noise_scale parameters are stored correctly."""
        dataset = UAAG2Dataset(DATA_PATH, mask_rate=mask_rate, pocket_noise=True, noise_scale=noise_scale)
        assert dataset.mask_rate == mask_rate
        assert dataset.pocket_noise is True
        assert dataset.noise_scale == noise_scale

    def test_charge_embedding_mapping(self):
        """Test that charge embedding dictionary is correctly initialized."""
        dataset = UAAG2Dataset(DATA_PATH)
        expected_charges = {-1: 0, 0: 1, 1: 2, 2: 3}
        assert dataset.charge_emb == expected_charges

    def test_invalid_path_raises_error(self):
        """Test that an error is raised when the data path doesn't exist."""
        with pytest.raises(lmdb.Error):
            UAAG2Dataset("/nonexistent/path/to/data.lmdb")
