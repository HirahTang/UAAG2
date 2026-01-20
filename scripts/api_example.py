#!/usr/bin/env python3
"""Example script demonstrating how to use the UAAG2 MNIST API.

This script shows how to make requests to the API endpoints.
Following DTU MLOps course module M22 (APIs) and M24 (API Testing).

Usage:
    python scripts/api_example.py
"""

from __future__ import annotations

import random

import requests


def main() -> None:
    """Demonstrate API usage."""
    # API base URL (change this if running on a different host)
    base_url = "http://localhost:8000"

    print("=" * 60)
    print("UAAG2 MNIST API Example")
    print("=" * 60)

    # 1. Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")

    # 2. Test health endpoint
    print("\n2. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")

    # 3. Test prediction endpoint with random data
    print("\n3. Testing prediction endpoint...")
    print("   Generating random MNIST-like image...")
    random.seed(42)
    data = [random.random() for _ in range(784)]

    response = requests.post(
        f"{base_url}/predict",
        json={"data": data},
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
    else:
        print(f"   Error: {response.text}")

    # 4. Test prediction with all zeros (blank image)
    print("\n4. Testing with blank image (all zeros)...")
    data = [0.0] * 784
    response = requests.post(
        f"{base_url}/predict",
        json={"data": data},
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")

    # 5. Test prediction with all ones (white image)
    print("\n5. Testing with white image (all ones)...")
    data = [1.0] * 784
    response = requests.post(
        f"{base_url}/predict",
        json={"data": data},
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")

    # 6. Test error handling
    print("\n6. Testing error handling (invalid input)...")
    data = [0.5] * 100  # Only 100 values instead of 784
    response = requests.post(
        f"{base_url}/predict",
        json={"data": data},
    )
    print(f"   Status: {response.status_code}")
    print(f"   Expected error (422 Validation Error): {response.status_code == 422}")

    print("\n" + "=" * 60)
    print("API example complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.ConnectionError:
        print("\n❌ Error: Could not connect to the API.")
        print("   Make sure the API is running on http://localhost:8000")
        print("   Start it with: uvicorn uaag2.api:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
