# UAAG2 API Dockerfile
# Build: docker build -f dockerfiles/api.dockerfile -t uaag2-api .
# Run:   docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models uaag2-api

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
# Copy project files
COPY README.md README.md
COPY LICENSE LICENSE
COPY src src/
COPY models/good_model/last.ckpt models/good_model/last.ckpt
COPY data/statistic.pkl data/statistic.pkl

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Expose the API port
EXPOSE 8080

# Run the FastAPI server with uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "uaag2.api:app", "--host", "0.0.0.0", "--port", "8080"]
