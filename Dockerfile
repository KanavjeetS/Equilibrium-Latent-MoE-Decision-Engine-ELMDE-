# ==============================================================================
# Production Dockerfile for Equilibrium Latent-MoE Decision Engine (ELMDE)
# ==============================================================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml .
COPY elmde/ ./elmde/
COPY checkpoints/ ./checkpoints/

# Install python dependencies
RUN uv pip install --system torch numpy scipy fastapi uvicorn pydantic httpx transformers prometheus-client

# Create non-root user for security
RUN useradd -m -u 1000 elmdeuser && \
    chown -R elmdeuser:elmdeuser /app
USER elmdeuser

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production multi-worker uvicorn server
CMD ["uvicorn", "elmde.serving:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
