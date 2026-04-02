# UAAG2 Evaluation/Sampling Dockerfile (GPU/CUDA)
# Build: docker build -f dockerfiles/evaluate_gpu.dockerfile -t uaag2-evaluate-gpu .
# Run:   docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/Samples:/app/Samples uaag2-evaluate-gpu [args]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3.12 \
        python3.12-venv \
        python3-pip \
        build-essential \
        gcc \
        g++ \
        libxrender1 \
        libxext6 \
        libgomp1 \
        git \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

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
COPY src src/

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Create necessary directories for outputs
RUN mkdir -p models data Samples ProteinGymSampling

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default entrypoint runs the evaluation script
ENTRYPOINT ["uv", "run", "python", "-m", "uaag2.evaluate"]
