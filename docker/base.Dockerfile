# =============================================================================
# Base Image - Minimal Python with core dependencies
# =============================================================================
# This image provides the foundation for all services with basic API capabilities
# Use case: API Gateway, simple services without ML dependencies
# =============================================================================

# Use our custom base image instead of repeating setup
FROM ir-python-base:latest AS base

# =============================================================================
# Builder Stage - Install Python dependencies
# =============================================================================
FROM base AS builder

# Copy requirements
COPY requirements/base.txt /tmp/requirements.txt

# Install dependencies to a separate location
RUN pip install --user --no-warn-script-location --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Runtime Stage - Final minimal image
# =============================================================================
FROM base AS runtime

# Copy installed dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy common code (shared modules)
COPY --chown=appuser:appuser common/ /app/common/

# Copy configuration files
COPY --chown=appuser:appuser configs/ /app/configs/

# Copy services code (needed for all services)
COPY --chown=appuser:appuser services/ /app/services/

# Copy scripts (needed for tools)
COPY --chown=appuser:appuser scripts/ /app/scripts/

# Set PATH for user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Health check - checks if FastAPI is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (override in docker-compose)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
