# =============================================================================
# ML Image - Python with ML/NLP dependencies and preloaded models
# =============================================================================
# This image includes PyTorch, Transformers, and preloaded SentenceTransformer models
# Use case: Retrieval, Ranking, Query Refinement services
# =============================================================================

FROM ir-python-base:latest AS base

# Set ML-specific environment variables
ENV TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models \
    NLTK_DATA=/app/.cache/nltk_data \
    SPACY_DATA=/app/.cache/spacy_data

# =============================================================================
# Dependencies Builder Stage
# =============================================================================
FROM base AS deps-builder

# Copy requirements
COPY requirements/base.txt requirements/ml.txt requirements/nlp.txt requirements/ir.txt /tmp/

# Install base dependencies first (without PyTorch index)
RUN pip install --user --no-warn-script-location --no-cache-dir \
    -r /tmp/base.txt \
    -r /tmp/nlp.txt \
    -r /tmp/ir.txt

# Install ML dependencies with PyTorch CPU index
RUN pip install --user --no-warn-script-location --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0 \
    transformers==4.40.2 \
    sentence-transformers==2.7.0 \
    scikit-learn==1.4.2 \
    scipy==1.13.0

# =============================================================================
# Models Download Stage - Download and cache models
# =============================================================================
FROM base AS models-downloader

# Copy dependencies from builder
COPY --from=deps-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    CUDA_VISIBLE_DEVICES=-1

# Create cache directories
RUN mkdir -p /app/data/models \
    /app/.cache/nltk_data \
    /app/.cache/spacy_data \
    /app/.cache/transformers \
    /app/.cache/torch

# Copy local model if available (for offline use)
# Note: Models will be loaded from volume mount at runtime (data/models)
# This allows using local models without building them into the image

# Download spaCy model (it installs to site-packages, not user directory)
RUN python -m spacy download en_core_web_sm || echo "spaCy model download failed, continuing..."

# Download NLTK data
RUN python -c "import nltk; nltk.download('wordnet', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('stopwords', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('omw-1.4', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('punkt', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('punkt_tab', download_dir='/app/.cache/nltk_data', quiet=True); print('NLTK data downloaded successfully')"

# Try to download SentenceTransformer model from HuggingFace, but continue if it fails (use local copy)
RUN python -c "import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'; from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu', trust_remote_code=False); print(f'Model cached successfully')" || echo "SentenceTransformer download failed, will use local model"

# =============================================================================
# Runtime Stage - Final optimized image
# =============================================================================
FROM base AS runtime

# Copy Python dependencies (includes spaCy model in site-packages)
COPY --from=deps-builder /root/.local /home/appuser/.local
# Copy spaCy model from system site-packages to user directory with correct ownership
COPY --from=models-downloader --chown=appuser:appuser /usr/local/lib/python3.11/site-packages/en_core_web_sm /home/appuser/.local/lib/python3.11/site-packages/en_core_web_sm
COPY --from=models-downloader --chown=appuser:appuser /usr/local/lib/python3.11/site-packages/en_core_web_sm-*.dist-info /home/appuser/.local/lib/python3.11/site-packages/

# Copy downloaded models and data
COPY --from=models-downloader --chown=appuser:appuser /app/.cache /app/.cache
# Note: /app/data/models will be mounted as a volume at runtime

# Copy common code
COPY --chown=appuser:appuser common/ /app/common/

# Copy configuration files
COPY --chown=appuser:appuser configs/ /app/configs/

# Copy services code
COPY --chown=appuser:appuser services/ /app/services/

# Copy scripts (for tools)
COPY --chown=appuser:appuser scripts/ /app/scripts/

# Create directories for runtime data (these will be mounted as volumes)
RUN mkdir -p /app/data/indices \
    /app/data/datasets \
    /app/data/models \
    /app/data/processed \
    /app/data/evaluation \
    /app/logs && \
    chown -R appuser:appuser /app/data /app/logs

# Set PATH and disable CUDA at runtime
ENV PATH=/home/appuser/.local/bin:$PATH \
    CUDA_VISIBLE_DEVICES=-1

# Switch to non-root user
USER appuser

# Verify installations (CPU-only)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" && \
    python -c "import sentence_transformers; print('SentenceTransformers OK')" && \
    python -c "import nltk; print('NLTK OK')" && \
    python -c "import spacy; print('spaCy OK')"

# Health check with longer start period for ML services
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
