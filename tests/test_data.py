import os
from dataclasses import dataclass
from torch.utils.data import Dataset

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

    def test_dataset_mask_rate(self):
        """Test that mask_rate parameter is stored correctly."""
        dataset = UAAG2Dataset(DATA_PATH, mask_rate=0.5)
        assert dataset.mask_rate == 0.5

    def test_dataset_pocket_noise(self):
        """Test that pocket_noise parameter is stored correctly."""
        dataset = UAAG2Dataset(DATA_PATH, pocket_noise=True, noise_scale=0.2)
        assert dataset.pocket_noise is True
        assert dataset.noise_scale == 0.2

    def test_charge_embedding_mapping(self):
        """Test that charge embedding dictionary is correctly initialized."""
        dataset = UAAG2Dataset(DATA_PATH)
        expected_charges = {-1: 0, 0: 1, 1: 2, 2: 3}
        assert dataset.charge_emb == expected_charges
