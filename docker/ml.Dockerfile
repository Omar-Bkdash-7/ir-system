# =============================================================================
# ML Image - Python with ML/NLP dependencies and preloaded models
# =============================================================================
# This image includes PyTorch, Transformers, and preloaded SentenceTransformer models
# Use case: Retrieval, Ranking, Query Refinement services
# =============================================================================

FROM ir-python-base:latest as base

# Set ML-specific environment variables
ENV TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    NLTK_DATA=/app/.cache/nltk_data \
    SPACY_DATA=/app/.cache/spacy_data

# =============================================================================
# Dependencies Builder Stage
# =============================================================================
FROM base as deps-builder

# Copy requirements
COPY requirements/base.txt requirements/ml.txt requirements/nlp.txt requirements/ir.txt /tmp/

# Install Python dependencies
RUN pip install --user --no-warn-script-location \
    -r /tmp/base.txt \
    -r /tmp/ml.txt \
    -r /tmp/nlp.txt \
    -r /tmp/ir.txt

# =============================================================================
# Models Download Stage - Download and cache models
# =============================================================================
FROM base as models-downloader

# Copy dependencies from builder
COPY --from=deps-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Create cache directories
RUN mkdir -p /app/.cache/sentence-transformers \
    /app/.cache/nltk_data \
    /app/.cache/spacy_data \
    /app/.cache/transformers \
    /app/.cache/torch

# Download spaCy model
RUN python -m spacy download en_core_web_sm && \
    mv /root/.local/lib/python3.11/site-packages/en_core_web_sm* /app/.cache/spacy_data/

# Download NLTK data
RUN python -c " \
import nltk; \
nltk.download('wordnet', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.download('stopwords', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.download('omw-1.4', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.download('punkt', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.download('punkt_tab', download_dir='/app/.cache/nltk_data', quiet=True); \
print('NLTK data downloaded successfully') \
"

# Download SentenceTransformer model
# This downloads the model into SENTENCE_TRANSFORMERS_HOME
RUN python -c " \
from sentence_transformers import SentenceTransformer; \
import os; \
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/.cache/sentence-transformers'; \
model = SentenceTransformer('all-MiniLM-L6-v2'); \
print(f'Model cached at: {os.environ[\"SENTENCE_TRANSFORMERS_HOME\"]}') \
"

# =============================================================================
# Runtime Stage - Final optimized image
# =============================================================================
FROM base as runtime

# Copy Python dependencies
COPY --from=deps-builder /root/.local /home/appuser/.local

# Copy downloaded models and data
COPY --from=models-downloader --chown=appuser:appuser /app/.cache /app/.cache

# Copy common code
COPY --chown=appuser:appuser common/ /app/common/

# Create directories for runtime data (these will be mounted as volumes)
RUN mkdir -p /app/data/indices \
    /app/data/datasets \
    /app/data/processed \
    /app/data/evaluation \
    /app/logs && \
    chown -R appuser:appuser /app/data /app/logs

# Set PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Verify installations
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" && \
    python -c "import sentence_transformers; print('SentenceTransformers OK')" && \
    python -c "import nltk; print('NLTK OK')" && \
    python -c "import spacy; print('spaCy OK')"

# Health check with longer start period for ML services
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
