# =============================================================================
# Simple Service Image - For services without ML dependencies
# =============================================================================
# Use case: API Gateway, Preprocessing (if no ML needed)
# Build: docker build -f docker/service-simple.Dockerfile -t ir-service-simple .
# Size: ~250MB
# =============================================================================

FROM ir-python-base:latest AS base

# =============================================================================
# Builder Stage
# =============================================================================
FROM base AS builder

# Accept build argument for specific requirements
ARG REQUIREMENTS_FILE=base.txt

COPY requirements/${REQUIREMENTS_FILE} /tmp/requirements.txt
RUN pip install --user --no-warn-script-location --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Runtime Stage
# =============================================================================
FROM base AS runtime

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser common/ /app/common/

# Copy configuration files
COPY --chown=appuser:appuser configs/ /app/configs/

# Copy services and scripts
COPY --chown=appuser:appuser services/ /app/services/
COPY --chown=appuser:appuser scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
