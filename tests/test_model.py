import torch
from uaag2.mnist_model import MyAwesomeModel


def test_model_construction():
    """Test that the model can be instantiated."""
    model = MyAwesomeModel()
    assert model is not None


def test_model_forward_pass():
    """Test that the model can perform a forward pass."""
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)
