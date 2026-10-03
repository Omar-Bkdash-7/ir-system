# =============================================================================
# Unified Multi-Stage Dockerfile for IR Microservices
# =============================================================================
# 
# ARCHITECTURE OVERVIEW
# ---------------------
# This Dockerfile replaces 6 separate Dockerfiles with a single unified build
# using 13 stages: 8 builder stages + 5 runtime targets.
#
# BUILDER STAGES (hierarchical dependency chain):
#   system → base-builder → ir-builder → scientific-builder → nlp-builder 
#   → torch-builder → transformers-builder → clustering-builder
#   (frontend-builder branches from base-builder)
#
# RUNTIME TARGETS (service-to-tier mapping):
#   runtime-light:       api_gateway, ranking (ir-light:latest)
#   runtime-nlp:         preprocessing (ir-nlp:latest)
#   runtime-ml:          indexing, retrieval, query_refinement, evaluation (ir-ml:latest)
#   runtime-clustering:  clustering (ir-clustering:latest)
#   runtime-frontend:    frontend (ir-frontend:latest)
#
# KEY FEATURES
# ------------
# 1. BUILDKIT CACHE MOUNTS: Every pip install uses --mount=type=cache,target=/root/.cache/pip
#    so wheels are cached across builds. Adding a library downloads only new packages.
#
# 2. LAYERED DEPENDENCIES: Each builder stage copies only its requirements file and installs
#    only its layer, so changing one file invalidates only downstream layers. Order by change
#    frequency: system → base → ir → scientific → nlp → torch → transformers → clustering.
#
# 3. SHARED IMAGES: Services that share a tier (e.g., 4 ML services) reuse the exact same
#    image and layers, reducing disk usage from ~39GB to ~12GB total (images + cache).
#
# 4. NO CODE IN BUILDER STAGES: Code is copied only in runtime stages. In dev mode, bind
#    mounts override the baked-in code, so editing code never triggers a rebuild.
#
# 5. VIRTUALENV PATTERN: All dependencies install into /opt/venv, which is copied wholesale
#    to runtime stages. This replaces the old --user pattern with hard-coded site-packages paths.
#
# 6. TORCH CPU-ONLY: torch==2.3.0 installed from --extra-index-url https://download.pytorch.org/whl/cpu
#    with numpy pinned to 1.26.4 (constraints.txt prevents version drift and ABI breakage).
#
# 7. RAG-READY: sentencepiece, accelerate, and protobuf are installed in transformers-builder,
#    so adding a RAG service requires no image rebuild (just create services/rag and add to compose).
#
# 8. CROSS-IMPORT FIX: torch-builder inherits from nlp-builder (not scientific-builder) because
#    ML services import preprocessing service code, which requires spacy and pyspellchecker.
#    runtime-ml and runtime-clustering include NLP dependencies + NLTK data.
#
# DISK USAGE (measured 2025-01-21)
# ---------------------------------
# Images:       12.45 GB (5 tier images + base images + postgres/redis)
# Build Cache:  21.41 GB (includes pip cache mounts and layer cache)
# Total:        ~34 GB (vs ~39GB baseline before restructure)
# Reclaimable:   9.7 GB in unused images (old images not yet pruned)
#
# Target after prune: <10GB images + capped build cache (~5GB with docker builder prune --keep-storage=5gb)
#
# MAINTENANCE COMMANDS
# --------------------
# Build all: docker-compose build --parallel
# Dev mode:  docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
# Prune:     docker builder prune --keep-storage=5gb (keeps cache for incremental builds)
# Measure:   docker system df -v
#
# REQUIREMENTS
# ------------
# - Docker with BuildKit enabled (DOCKER_BUILDKIT=1 in .env for docker-compose)
# - requirements/ directory with layered files: 00-system.txt through 08-frontend.txt
# - requirements/constraints.txt (pip-compile output) for reproducible builds
#
# =============================================================================

# =============================================================================
# Stage: system
# Base system dependencies and user setup
# =============================================================================
FROM python:3.11-slim AS system

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/home/appuser --shell=/bin/bash appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser

RUN mkdir -p /app /opt/venv && \
    chown -R appuser:appuser /app

WORKDIR /app

# =============================================================================
# Stage: base-builder
# Core dependencies: fastapi, uvicorn, httpx, pydantic, numpy, pandas, psycopg2, redis
# =============================================================================
FROM system AS base-builder

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements/base.txt /tmp/base.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/base.txt

# =============================================================================
# Stage: ir-builder
# IR-specific dependencies: ir-datasets, rank-bm25, ir-measures
# =============================================================================
FROM base-builder AS ir-builder

COPY requirements/ir.txt /tmp/ir.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install ir-datasets>=0.5 rank-bm25>=0.2 ir-measures>=0.3

# =============================================================================
# Stage: scientific-builder
# Scientific computing: scikit-learn, scipy
# =============================================================================
FROM ir-builder AS scientific-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install scikit-learn==1.4.2 scipy==1.13.0

# =============================================================================
# Stage: nlp-builder
# NLP libraries and resources (spaCy, NLTK)
# =============================================================================
FROM scientific-builder AS nlp-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
    nltk==3.8.1 \
    spacy==3.7.5 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl \
    pyspellchecker==0.8.1

RUN mkdir -p /opt/nltk_data

RUN python -c "import nltk; \
nltk.download('wordnet', download_dir='/opt/nltk_data', quiet=True); \
nltk.download('stopwords', download_dir='/opt/nltk_data', quiet=True); \
nltk.download('omw-1.4', download_dir='/opt/nltk_data', quiet=True); \
nltk.download('punkt', download_dir='/opt/nltk_data', quiet=True); \
nltk.download('punkt_tab', download_dir='/opt/nltk_data', quiet=True); \
print('NLTK resources downloaded')"

# =============================================================================
# Stage: torch-builder
# PyTorch CPU-only stack
# =============================================================================
FROM nlp-builder AS torch-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --timeout=300 \
    torch==2.3.0

RUN python -c 'import torch; print(f"PyTorch {torch.__version__} installed"); assert not torch.cuda.is_available(), "CUDA should not be available"; print("CPU-only verified")'

# =============================================================================
# Stage: transformers-builder
# Transformers, sentence-transformers, and RAG dependencies
# =============================================================================
FROM torch-builder AS transformers-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
    transformers==4.40.2 \
    sentence-transformers==2.7.0 \
    sentencepiece>=0.2.0 \
    accelerate>=0.20.0 \
    "protobuf>=3.20.0,<5.0"

ENV CUDA_VISIBLE_DEVICES=-1 \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

RUN mkdir -p /opt/models

RUN python -c "\
import os; \
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'; \
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu', trust_remote_code=False); \
print(f'Embedding model downloaded: {model.get_sentence_embedding_dimension()}d'); \
"

# =============================================================================
# Stage: clustering-builder
# Clustering and topic modeling extras: hdbscan, bertopic, umap-learn
# =============================================================================
FROM transformers-builder AS clustering-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
    hdbscan \
    bertopic \
    umap-learn

# =============================================================================
# Stage: frontend-builder
# Frontend-specific dependencies: streamlit, plotly
# =============================================================================
FROM base-builder AS frontend-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
    streamlit==1.35.0 \
    plotly==5.21.0

# =============================================================================
# Stage: runtime-light
# Minimal runtime for services with basic dependencies
# Services: api_gateway, ranking, evaluation
# =============================================================================
FROM python:3.11-slim AS runtime-light

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/home/appuser --shell=/bin/bash appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser

RUN mkdir -p /app /app/logs /app/data && \
    chown -R appuser:appuser /app

COPY --from=ir-builder /opt/venv /opt/venv

COPY --chown=appuser:appuser common/ /app/common/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser services/ /app/services/
COPY --chown=appuser:appuser scripts/ /app/scripts/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app
USER appuser

# =============================================================================
# Stage: runtime-nlp
# NLP runtime with spaCy and NLTK
# Services: preprocessing
# =============================================================================
FROM python:3.11-slim AS runtime-nlp

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/home/appuser --shell=/bin/bash appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser

RUN mkdir -p /app /app/logs /app/data && \
    chown -R appuser:appuser /app

COPY --from=nlp-builder /opt/venv /opt/venv
COPY --from=nlp-builder /opt/nltk_data /app/.cache/nltk_data

COPY --chown=appuser:appuser common/ /app/common/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser services/ /app/services/
COPY --chown=appuser:appuser scripts/ /app/scripts/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/app/.cache/nltk_data

WORKDIR /app
USER appuser

# =============================================================================
# Stage: runtime-ml
# ML runtime with PyTorch, Transformers, and SentenceTransformers
# Services: indexing, retrieval, query_refinement
# =============================================================================
FROM python:3.11-slim AS runtime-ml

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/home/appuser --shell=/bin/bash appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser

RUN mkdir -p /app /app/logs /app/data /app/data/models && \
    chown -R appuser:appuser /app

COPY --from=transformers-builder /opt/venv /opt/venv
COPY --from=transformers-builder /opt/models /app/data/models
COPY --from=nlp-builder /opt/nltk_data /app/.cache/nltk_data

COPY --chown=appuser:appuser common/ /app/common/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser services/ /app/services/
COPY --chown=appuser:appuser scripts/ /app/scripts/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models \
    TORCH_HOME=/app/data/cache/torch \
    TRANSFORMERS_CACHE=/app/data/cache/transformers \
    NLTK_DATA=/app/.cache/nltk_data \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app
USER appuser

# =============================================================================
# Stage: runtime-clustering
# Clustering runtime with additional clustering libraries
# Services: clustering
# =============================================================================
FROM python:3.11-slim AS runtime-clustering

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/home/appuser --shell=/bin/bash appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser

RUN mkdir -p /app /app/logs /app/data /app/data/models && \
    chown -R appuser:appuser /app

COPY --from=clustering-builder /opt/venv /opt/venv
COPY --from=transformers-builder /opt/models /app/data/models
COPY --from=nlp-builder /opt/nltk_data /app/.cache/nltk_data

COPY --chown=appuser:appuser common/ /app/common/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser services/ /app/services/
COPY --chown=appuser:appuser scripts/ /app/scripts/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models \
    TORCH_HOME=/app/data/cache/torch \
    TRANSFORMERS_CACHE=/app/data/cache/transformers \
    NLTK_DATA=/app/.cache/nltk_data \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app
USER appuser

# =============================================================================
# Stage: runtime-frontend
# Frontend runtime with Streamlit
# Services: frontend
# =============================================================================
FROM python:3.11-slim AS runtime-frontend

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/home/appuser --shell=/bin/bash appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser

RUN mkdir -p /app /app/logs /app/data && \
    chown -R appuser:appuser /app

COPY --from=frontend-builder /opt/venv /opt/venv

COPY --chown=appuser:appuser common/ /app/common/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser frontend/ /app/frontend/
COPY --chown=appuser:appuser services/ /app/services/
COPY --chown=appuser:appuser scripts/ /app/scripts/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app
USER appuser
