# =============================================================================
# Python Base Image - Foundation for ALL services
# =============================================================================
# This is the REAL base image that all other images will inherit from
# It contains common system dependencies and user setup
# 
# Build: docker build -f docker/python-base.Dockerfile -t ir-python-base:latest .
# Size: ~180MB
# 
# This eliminates duplication across all Dockerfiles and ensures consistency
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="IR System Team"
LABEL description="Base Python image for IR microservices"
LABEL version="1.0"

# =============================================================================
# Environment Variables - Common to ALL services
# =============================================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# =============================================================================
# System Dependencies - Common build tools
# =============================================================================
# These are needed for compiling Python packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    gcc \
    g++ \
    make \
    # Networking tools (for health checks and debugging)
    curl \
    wget \
    # Git (for some pip packages that need it)
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# Create non-root user and group
# =============================================================================
# Security best practice: never run as root in production
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/home/appuser --shell=/bin/bash appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser

# =============================================================================
# Create standard application directories
# =============================================================================
RUN mkdir -p /app /app/logs /app/data && \
    chown -R appuser:appuser /app

WORKDIR /app

# =============================================================================
# Health check script for services without HTTP endpoints
# =============================================================================
# This will be overridden in service-specific Dockerfiles
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (will be overridden)
CMD ["python", "--version"]
