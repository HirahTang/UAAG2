# UAAG2 Training Dockerfile (GPU/CUDA)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY README.md README.md
COPY LICENSE LICENSE
COPY tasks.py tasks.py

COPY configs configs/
COPY src src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN mkdir -p models data reports/figures 3DcoordsAtomsBonds_0

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["uv", "run", "invoke", "fetch-data", "train", "--num-epochs=1"]
