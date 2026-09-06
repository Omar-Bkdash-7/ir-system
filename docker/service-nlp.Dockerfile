# =============================================================================
# NLP Service Image - For preprocessing service
# =============================================================================
# Includes NLP libraries (spaCy, NLTK) but not heavy ML libraries
# Use case: Preprocessing service
# Size: ~800MB
# =============================================================================

FROM ir-python-base:latest AS base

ENV NLTK_DATA=/app/.cache/nltk_data \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models \
    TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_CACHE=/app/.cache/transformers

# =============================================================================
# Dependencies Builder
# =============================================================================
FROM base AS deps-builder

COPY requirements/base.txt requirements/nlp.txt /tmp/
RUN pip install --user --no-warn-script-location --no-cache-dir \
    -r /tmp/base.txt \
    -r /tmp/nlp.txt

# =============================================================================
# NLP Resources Downloader
# =============================================================================
FROM base AS nlp-downloader

COPY --from=deps-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

RUN mkdir -p /app/.cache/nltk_data

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('wordnet', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('stopwords', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('omw-1.4', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('punkt', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('punkt_tab', download_dir='/app/.cache/nltk_data', quiet=True); print('NLTK resources downloaded')"

# Download SentenceTransformer model (used by query_refinement service)
RUN python -c "\
import os; \
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'; \
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu', trust_remote_code=False); \
print(f'Embedding model downloaded: {model.get_sentence_embedding_dimension()}d'); \
" || echo "SentenceTransformer download failed, will use local model from volume mount"

# =============================================================================
# Runtime Stage
# =============================================================================
FROM base AS runtime

COPY --from=deps-builder /root/.local /home/appuser/.local
COPY --from=nlp-downloader --chown=appuser:appuser /app/.cache /app/.cache
# Copy spaCy model directory and its dist-info explicitly (same approach as ml.Dockerfile)
COPY --from=nlp-downloader --chown=appuser:appuser /usr/local/lib/python3.11/site-packages/en_core_web_sm /home/appuser/.local/lib/python3.11/site-packages/en_core_web_sm
COPY --from=nlp-downloader --chown=appuser:appuser /usr/local/lib/python3.11/site-packages/en_core_web_sm-3.8.0.dist-info /home/appuser/.local/lib/python3.11/site-packages/en_core_web_sm-3.8.0.dist-info
COPY --chown=appuser:appuser common/ /app/common/

# Copy configuration files
COPY --chown=appuser:appuser configs/ /app/configs/

# Copy services code
COPY --chown=appuser:appuser services/ /app/services/

# Copy scripts
COPY --chown=appuser:appuser scripts/ /app/scripts/

RUN mkdir -p /app/logs /app/data/processed && \
    chown -R appuser:appuser /app/logs /app/data

ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser

RUN python -c "import spacy; print('spaCy OK')" && \
    python -c "import nltk; print('NLTK OK')"

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "services.preprocessing.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
