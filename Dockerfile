# ASRI Backend - Production Dockerfile
# Multi-stage build for smaller image

FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency resolution
RUN pip install uv

# Copy all files needed for build (hatchling needs README.md)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install .

# Production stage
FROM python:3.12-slim AS production

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash asri

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Set ownership
RUN chown -R asri:asri /app

USER asri

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Run with uvicorn
CMD ["uvicorn", "asri.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
