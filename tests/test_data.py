from tests import _PATH_DATA

from torch.utils.data import Dataset

from uaag2.mnist_data import corrupt_mnist


# def test_normalize(images: torch.Tensor) -> torch.Tensor:
#     """Normalize images."""
#     return (images - images.mean()) / images.std()


# def test_preprocess_data(raw_dir: str, processed_dir: str) -> None:


# uvx invoke

def test_dataset():
    """Test loading the dataset."""
    train, test = corrupt_mnist()
    assert isinstance(train, Dataset)
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()
