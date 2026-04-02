# UAAG2 Training Dockerfile
# Build: docker build -f dockerfiles/train.dockerfile -t uaag2-train .
# Run:   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models uaag2-train [args]

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Install system dependencies required for scientific Python packages
# - build-essential: C/C++ compilers for building extensions
# - libxrender1, libxext6: Required by RDKit for molecule visualization
# - libgomp1: OpenMP support for parallel computation
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        g++ \
        libxrender1 \
        libxext6 \
        libgomp1 \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better layer caching
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Install dependencies without the project itself (better caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy project files
COPY README.md README.md
COPY LICENSE LICENSE
COPY tasks.py tasks.py
COPY configs configs/
COPY src src/

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Create necessary directories for outputs
RUN mkdir -p models data reports/figures 3DcoordsAtomsBonds_0

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default entrypoint runs the training script
# Users can override with additional arguments
ENV WANDB_MODE=offline
ENTRYPOINT ["uv", "run", "invoke", "fetch-data", "train", "--num-epochs=1", "--gpus=0"]
