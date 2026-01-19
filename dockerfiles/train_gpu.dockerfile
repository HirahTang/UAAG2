# UAAG2 Training Dockerfile (GPU/CUDA)
# Build: docker build -f dockerfiles/train_gpu.dockerfile -t uaag2-train-gpu .
# Run:   docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models uaag2-train-gpu [args]

# 1. Use NVIDIA CUDA 12.4 Base Image (Ubuntu 22.04)
# We use 'cudnn-runtime' to ensure all deep learning libs are present
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install System Deps
# We REMOVED python3/pip because 'uv' will download its own managed Python 3.12
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        g++ \
        libxrender1 \
        libxext6 \
        libgomp1 \
        git \
        curl \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Install uv manually
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 4. Standard Build Steps (Same as CPU file)
WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Install dependencies
# uv automatically fetches the Python 3.12 interpreter here
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY README.md README.md
COPY LICENSE LICENSE
COPY tasks.py tasks.py
COPY src src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN mkdir -p models data reports/figures 3DcoordsAtomsBonds_0

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 5. Runtime
ENTRYPOINT ["uv", "run", "invoke", "fetch-data", "train", "--num-epochs=1"]
