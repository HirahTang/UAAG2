"""API integration tests for UAAG2 MNIST inference API.

This module tests the FastAPI endpoints to ensure they work as intended.
Following DTU MLOps course module M24 (API Testing).
"""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from uaag2.api import app

# Create test client
client = TestClient(app)


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_read_root(self) -> None:
        """Test root endpoint returns welcome message.

        Following DTU MLOps course material - testing basic endpoint functionality.
        """
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the UAAG2 MNIST model inference API!"}

    def test_health_check(self) -> None:
        """Test health endpoint returns correct status.

        Following DTU MLOps best practices for health monitoring.
        """
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ["healthy", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)

    def test_predict_valid_input(self) -> None:
        """Test prediction endpoint with valid MNIST-like input.

        Following DTU MLOps course material - testing with realistic data.
        """
        # Create a random valid input (784 values between 0 and 1)
        random.seed(42)
        data = [random.random() for _ in range(784)]

        response = client.post("/predict", json={"data": data})
        assert response.status_code == 200

        result = response.json()
        assert "prediction" in result
        assert "confidence" in result
        assert 0 <= result["prediction"] <= 9
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_invalid_input_too_short(self) -> None:
        """Test prediction endpoint rejects input with insufficient data.

        Following DTU MLOps course material - testing error handling.
        """
        # Input with only 100 values (should be 784)
        data = [0.5] * 100

        response = client.post("/predict", json={"data": data})
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_input_too_long(self) -> None:
        """Test prediction endpoint rejects input with too much data.

        Following DTU MLOps course material - testing error handling.
        """
        # Input with 1000 values (should be 784)
        data = [0.5] * 1000

        response = client.post("/predict", json={"data": data})
        assert response.status_code == 422  # Validation error

    def test_predict_zeros_input(self) -> None:
        """Test prediction with all zeros (blank image).

        Following DTU MLOps course material - testing edge cases.
        """
        data = [0.0] * 784

        response = client.post("/predict", json={"data": data})
        assert response.status_code == 200

        result = response.json()
        assert 0 <= result["prediction"] <= 9
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_ones_input(self) -> None:
        """Test prediction with all ones (fully white image).

        Following DTU MLOps course material - testing edge cases.
        """
        data = [1.0] * 784

        response = client.post("/predict", json={"data": data})
        assert response.status_code == 200

        result = response.json()
        assert 0 <= result["prediction"] <= 9
        assert 0.0 <= result["confidence"] <= 1.0

    def test_multiple_predictions(self) -> None:
        """Test that multiple predictions work consistently.

        Following DTU MLOps course material - testing API reliability.
        """
        random.seed(123)
        for _ in range(5):
            data = [random.random() for _ in range(784)]
            response = client.post("/predict", json={"data": data})
            assert response.status_code == 200
            result = response.json()
            assert 0 <= result["prediction"] <= 9
            assert 0.0 <= result["confidence"] <= 1.0


class TestAPIValidation:
    """Test suite for input validation."""

    def test_predict_missing_data_field(self) -> None:
        """Test that missing data field is rejected.

        Following DTU MLOps course material - testing input validation.
        """
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_data_type(self) -> None:
        """Test that non-numeric data is rejected.

        Following DTU MLOps course material - testing input validation.
        """
        data = ["not", "a", "number"] * 262  # 786 elements, but strings
        response = client.post("/predict", json={"data": data})
        assert response.status_code == 422  # Validation error

    def test_predict_none_values(self) -> None:
        """Test that None values in data are rejected.

        Following DTU MLOps course material - testing input validation.
        """
        data = [None] * 784
        response = client.post("/predict", json={"data": data})
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
