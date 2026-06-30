# =============================================================================
# NLP Service Image - For preprocessing service
# =============================================================================
# Includes NLP libraries (spaCy, NLTK) but not heavy ML libraries
# Use case: Preprocessing service
# Size: ~800MB
# =============================================================================

FROM ir-python-base:latest as base

ENV NLTK_DATA=/app/.cache/nltk_data

# =============================================================================
# Dependencies Builder
# =============================================================================
FROM base as deps-builder

COPY requirements/base.txt requirements/nlp.txt /tmp/
RUN pip install --user --no-warn-script-location \
    -r /tmp/base.txt \
    -r /tmp/nlp.txt

# =============================================================================
# NLP Resources Downloader
# =============================================================================
FROM base as nlp-downloader

COPY --from=deps-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

RUN mkdir -p /app/.cache/nltk_data

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c " \
import nltk; \
nltk.download('wordnet', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.download('stopwords', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.download('omw-1.4', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.download('punkt', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.download('punkt_tab', download_dir='/app/.cache/nltk_data', quiet=True); \
print('NLTK resources downloaded') \
"

# =============================================================================
# Runtime Stage
# =============================================================================
FROM base as runtime

COPY --from=deps-builder /root/.local /home/appuser/.local
COPY --from=nlp-downloader --chown=appuser:appuser /app/.cache /app/.cache
COPY --chown=appuser:appuser common/ /app/common/

RUN mkdir -p /app/logs /app/data/processed && \
    chown -R appuser:appuser /app/logs /app/data

ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser

RUN python -c "import spacy; print('spaCy OK')" && \
    python -c "import nltk; print('NLTK OK')"

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
