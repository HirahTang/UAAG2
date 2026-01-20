"""FastAPI application for UAAG2 MNIST model inference.

This module provides a REST API for serving predictions from the MNIST model.
Following DTU MLOps course module M22 (APIs) and M24 (API Testing).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from uaag2.mnist_model import MyAwesomeModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model: MyAwesomeModel | None = None


class PredictRequest(BaseModel):
    """Request model for prediction endpoint.

    Attributes:
        data: A flattened list of 784 pixel values (28x28 image) normalized between 0 and 1.
    """

    data: list[float] = Field(
        ...,
        description="Flattened MNIST image as list of 784 values",
        min_length=784,
        max_length=784,
    )


class PredictResponse(BaseModel):
    """Response model for prediction endpoint.

    Attributes:
        prediction: The predicted digit (0-9).
        confidence: The confidence score of the prediction.
    """

    prediction: int = Field(..., ge=0, le=9, description="Predicted digit (0-9)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")


class HealthResponse(BaseModel):
    """Response model for health check endpoint.

    Attributes:
        status: The health status of the API.
        model_loaded: Whether the model is loaded and ready.
    """

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events.

    Loads the model at startup and cleans up at shutdown.
    Following DTU MLOps best practices for model loading.
    """
    global model
    model_path = os.getenv("MODEL_PATH", "models/model.pth")

    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model = MyAwesomeModel()
            state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}, using untrained model for testing")
            model = MyAwesomeModel()
            model.eval()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = MyAwesomeModel()
        model.eval()
        logger.warning("Using untrained model due to loading error")

    yield

    # Cleanup
    model = None
    logger.info("Model unloaded")


# Create FastAPI app with lifespan
app = FastAPI(
    title="UAAG2 MNIST Inference API",
    description="API for MNIST digit classification using the UAAG2 MLOps pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=dict[str, str])
async def root() -> dict[str, str]:
    """Root endpoint returning welcome message.

    Returns:
        dict: Welcome message.
    """
    return {"message": "Welcome to the UAAG2 MNIST model inference API!"}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse: Current health status and model load status.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictResponse, status_code=HTTPStatus.OK)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict digit from MNIST image data.

    Args:
        request: Prediction request containing image data.

    Returns:
        PredictResponse: Prediction result with digit and confidence.

    Raises:
        HTTPException: If model is not loaded or prediction fails.
    """
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Convert input to tensor and reshape to (1, 1, 28, 28)
        input_tensor = torch.tensor(request.data, dtype=torch.float32).view(1, 1, 28, 28)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()

        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
