from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from uaag2.api import app

client = TestClient(app)


def test_read_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("uaag2.api.MODEL")
@patch("uaag2.api.Dataset_Info")
@patch("uaag2.api.UAAG2Dataset_sampling")
@patch("uaag2.api.DataLoader")
def test_generate_mocked(mock_dataloader, mock_dataset, mock_dataset_info, mock_model):
    # Mock MODEL
    mock_model_instance = MagicMock()
    mock_model_instance.save_dir = "/tmp/mock_save_dir"
    mock_model.return_value = mock_model_instance

    # Needs to be set on the module level because it is a global variable
    with patch("uaag2.api.MODEL", mock_model_instance):
        # Create a mock file
        file_content = b"fake content"

        # We also need to mock torch.load because the endpoint loads the file
        with patch("torch.load") as mock_torch_load:
            # Mock the return value of torch.load to be a list containing a Data object
            mock_data = MagicMock()
            mock_torch_load.return_value = [mock_data]

            # Mock os.path.exists to return True for result files check
            with patch("os.path.exists") as mock_exists:
                # We want to simulate that the output file exists
                def side_effect(path):
                    if "ligand.mol" in path:
                        return True
                    if "statistic.pkl" in path:  # For load_data_info if it's called, but we mocked DATASET_INFO?
                        return True
                    return False

                mock_exists.side_effect = side_effect

                # Mock open to return dummy content
                with patch("builtins.open", new_callable=MagicMock) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = "mock molecule content"

                    response = client.post(
                        "/generate", files={"file": ("test.pt", file_content, "application/octet-stream")}
                    )

    # Since we are mocking quite a bit, we mainly expect 200 OK and some result structure
    # However, the endpoint code is complex with many side effects (saving files, etc.)
    # Let's adjust expectations. If we mock everything correctly, we get results.

    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) > 0
    assert response.json()["results"][0]["content"] == "mock molecule content"
