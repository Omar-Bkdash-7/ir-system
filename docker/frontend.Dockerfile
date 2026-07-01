# =============================================================================
# Frontend Image - Streamlit UI
# =============================================================================
# Lightweight image for Streamlit dashboard
# Size: ~500MB
# =============================================================================

FROM ir-python-base:latest AS base

# =============================================================================
# Builder Stage
# =============================================================================
FROM base AS builder

COPY requirements/base.txt requirements/frontend.txt /tmp/
RUN pip install --user --no-warn-script-location --no-cache-dir -r /tmp/frontend.txt

# =============================================================================
# Runtime Stage
# =============================================================================
FROM base AS runtime

# Copy dependencies
COPY --from=builder /root/.local /home/appuser/.local

# Copy common code
COPY --chown=appuser:appuser common/ /app/common/

# Copy configuration files
COPY --chown=appuser:appuser configs/ /app/configs/

# Copy .streamlit config
COPY --chown=appuser:appuser .streamlit/ /app/.streamlit/

# Copy services code
COPY --chown=appuser:appuser services/ /app/services/

# Copy scripts
COPY --chown=appuser:appuser scripts/ /app/scripts/

# Copy frontend code
COPY --chown=appuser:appuser frontend/ /app/frontend/

ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser

# Streamlit specific configuration
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "frontend/app.py", "--server.fileWatcherType=poll"]
