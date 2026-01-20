"""Locust load testing file for UAAG2 MNIST inference API.

This module defines load tests for the API endpoints using Locust.
Following DTU MLOps course module M24 (Load Testing).

Usage:
    # Web UI mode:
    locust -f tests/performancetests/locustfile.py

    # Headless mode:
    locust -f tests/performancetests/locustfile.py \
        --headless --users 10 --spawn-rate 1 --run-time 1m --host http://localhost:8000
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


class MNISTAPIUser(HttpUser):
    """A Locust user class simulating users interacting with the MNIST API.

    This class defines the tasks that simulated users will perform on the API.
    Following DTU MLOps course material on load testing with Locust.

    Attributes:
        wait_time: Time to wait between tasks (1-3 seconds).
    """

    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Called when a simulated user starts.

        Following DTU MLOps best practices for initialization.
        """
        # Pre-generate some test data for efficiency
        random.seed()
        self.sample_inputs = [
            [random.random() for _ in range(784)] for _ in range(10)
        ]

    @task(1)
    def get_root(self) -> None:
        """Task: Visit the root endpoint.

        Simulates a user checking if the API is alive.
        Following DTU MLOps course material - basic endpoint testing.
        """
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(2)
    def get_health(self) -> None:
        """Task: Check the health endpoint.

        Simulates monitoring services checking API health.
        Following DTU MLOps best practices for health monitoring.
        Weight: 2 (twice as likely as root endpoint).
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"API is unhealthy: {data}")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(10)
    def post_predict(self) -> None:
        """Task: Make a prediction request.

        Simulates users requesting predictions for MNIST images.
        Following DTU MLOps course material - testing main functionality.
        Weight: 10 (most common task, main API function).
        """
        # Use pre-generated sample input for efficiency
        data = random.choice(self.sample_inputs)

        with self.client.post(
            "/predict",
            json={"data": data},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                result = response.json()
                # Validate response structure
                if "prediction" in result and "confidence" in result:
                    if 0 <= result["prediction"] <= 9 and 0.0 <= result["confidence"] <= 1.0:
                        response.success()
                    else:
                        response.failure(f"Invalid prediction values: {result}")
                else:
                    response.failure(f"Missing fields in response: {result}")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def post_predict_edge_case_zeros(self) -> None:
        """Task: Test prediction with all zeros.

        Simulates edge case testing with blank images.
        Following DTU MLOps course material - testing edge cases.
        """
        data = [0.0] * 784

        with self.client.post(
            "/predict",
            json={"data": data},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Edge case failed with status {response.status_code}")

    @task(1)
    def post_predict_edge_case_ones(self) -> None:
        """Task: Test prediction with all ones.

        Simulates edge case testing with fully white images.
        Following DTU MLOps course material - testing edge cases.
        """
        data = [1.0] * 784

        with self.client.post(
            "/predict",
            json={"data": data},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Edge case failed with status {response.status_code}")


class StressTestUser(HttpUser):
    """A more aggressive user for stress testing.

    This class simulates high-frequency users for stress testing the API.
    Following DTU MLOps course material - identifying bottlenecks.

    Attributes:
        wait_time: Minimal wait time between requests (0.1-0.5 seconds).
    """

    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        """Initialize with pre-generated test data."""
        random.seed()
        self.sample_data = [random.random() for _ in range(784)]

    @task
    def rapid_predict(self) -> None:
        """Task: Rapid-fire prediction requests.

        Simulates high-load scenarios to find performance bottlenecks.
        Following DTU MLOps course material - performance testing.
        """
        self.client.post("/predict", json={"data": self.sample_data})


if __name__ == "__main__":
    # This allows running the file directly for testing
    import os
    import subprocess

    # Set default host if not already set
    if "MYENDPOINT" not in os.environ:
        os.environ["MYENDPOINT"] = "http://localhost:8000"

    # Run locust with default parameters
    subprocess.run(
        [
            "locust",
            "-f",
            __file__,
            "--headless",
            "--users",
            "10",
            "--spawn-rate",
            "1",
            "--run-time",
            "30s",
            "--host",
            os.environ["MYENDPOINT"],
        ]
    )
