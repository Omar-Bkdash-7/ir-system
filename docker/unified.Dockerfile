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
#   runtime-rag:         rag (ir-rag:latest)
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
# Stage: runtime-base
# Everything the six runtime targets share: curl (used by the compose healthchecks),
# the unprivileged user and the writable application directories. Defined once here
# instead of being copied into every runtime stage.
# =============================================================================
FROM python:3.11-slim AS runtime-base

RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/home/appuser --shell=/bin/bash appuser && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser && \
    mkdir -p /app /app/logs /app/data && \
    chown -R appuser:appuser /app

WORKDIR /app

# =============================================================================
# Stage: base-builder
# Core dependencies: fastapi, uvicorn, httpx, pydantic, numpy, pandas, psycopg2, redis
# =============================================================================
FROM system AS base-builder

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements/*.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/01-base.txt

# Test tooling (pytest) lives in the shared base so every runtime image can run the suite as-is.
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/10-dev.txt

# =============================================================================
# Stage: ir-builder
# IR-specific dependencies: ir-datasets, rank-bm25, ir-measures
# =============================================================================
FROM base-builder AS ir-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/07-ir.txt

# =============================================================================
# Stage: scientific-builder
# Scientific computing: scikit-learn, scipy
# =============================================================================
FROM ir-builder AS scientific-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/02-scientific.txt

# =============================================================================
# Stage: nlp-builder
# NLP libraries and resources (spaCy, NLTK)
# =============================================================================
FROM scientific-builder AS nlp-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/03-nlp.txt

RUN mkdir -p /opt/nltk_data

# NLTK corpora are fetched once, at build time, and the archives are verified.
# A download can return an HTML error page with HTTP 200 and nltk happily writes it to
# corpora/<name>.zip; that file only fails later, at request time. wordnet and stopwords
# are used by the pipeline, so a broken archive must stop the build; omw-1.4 only adds
# multilingual lemmas (English lemmas come from wordnet) and its upstream archive has
# been serving non-zip bodies, so it is kept when usable and dropped otherwise.
RUN python - <<'PY'
import os, shutil, sys, time, zipfile

import nltk

DATA = "/opt/nltk_data"
REQUIRED = ("wordnet", "stopwords")
OPTIONAL = ("omw-1.4",)
NO_VERIFY = ("punkt", "punkt_tab")


def fetch(name, attempts=3):
    for attempt in range(1, attempts + 1):
        if nltk.download(name, download_dir=DATA, quiet=True):
            return True
        print(f"{name}: attempt {attempt} failed", flush=True)
        time.sleep(2 * attempt)
    print(f"{name}: giving up after {attempts} attempts", flush=True)
    return False


def archive_ok(path):
    try:
        with zipfile.ZipFile(path) as bundle:
            if not bundle.namelist():
                return False, "archive is empty"
        return True, ""
    except Exception as exc:
        return False, str(exc)


for name in REQUIRED + OPTIONAL + NO_VERIFY:
    fetch(name)

for name in REQUIRED:
    path = f"{DATA}/corpora/{name}.zip"
    ok, why = archive_ok(path)
    if not ok:
        sys.exit(f"NLTK corpus {name!r} is unusable after download ({path}): {why}")
    print(f"NLTK corpus {name!r} verified", flush=True)

for name in OPTIONAL:
    path = f"{DATA}/corpora/{name}.zip"
    ok, why = archive_ok(path)
    if ok:
        print(f"NLTK corpus {name!r} verified", flush=True)
        continue
    print(f"NLTK corpus {name!r} unusable ({why}); dropping it", flush=True)
    shutil.rmtree(f"{DATA}/corpora/{name}", ignore_errors=True)
    try:
        os.remove(path)
    except OSError:
        pass
PY

# =============================================================================
# Stage: torch-builder
# PyTorch CPU-only stack
# =============================================================================
FROM nlp-builder AS torch-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --timeout=300 \
    -r /tmp/04-torch.txt

RUN python -c 'import torch; print(f"PyTorch {torch.__version__} installed"); assert not torch.cuda.is_available(), "CUDA should not be available"; print("CPU-only verified")'

# =============================================================================
# Stage: transformers-builder
# Transformers, sentence-transformers, and RAG dependencies
# =============================================================================
FROM torch-builder AS transformers-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/05-transformers.txt

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
# Clustering extras: hdbscan + umap-learn (topics use c-TF-IDF from scikit-learn, see services/clustering)
# =============================================================================
FROM transformers-builder AS clustering-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/06-clustering.txt

# =============================================================================
# Stage: rag-builder
# RAG (Retrieval-Augmented Generation) dependencies: langchain, openai, faiss
# =============================================================================
FROM transformers-builder AS rag-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/09-rag.txt

# =============================================================================
# Stage: frontend-builder
# Frontend-specific dependencies: streamlit, plotly
# =============================================================================
FROM base-builder AS frontend-builder

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/08-frontend.txt

# =============================================================================
# Stage: runtime-light
# Minimal runtime for services with basic dependencies
# Services: api_gateway, ranking, evaluation
# =============================================================================
FROM runtime-base AS runtime-light

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
FROM runtime-base AS runtime-nlp

COPY --from=nlp-builder /opt/venv /opt/venv
COPY --chown=appuser:appuser --from=nlp-builder /opt/nltk_data /app/.cache/nltk_data

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
FROM runtime-base AS runtime-ml

RUN mkdir -p /app/data/models && chown -R appuser:appuser /app/data/models

COPY --from=transformers-builder /opt/venv /opt/venv
COPY --chown=appuser:appuser --from=nlp-builder /opt/nltk_data /app/.cache/nltk_data

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
FROM runtime-base AS runtime-clustering

RUN mkdir -p /app/data/models && chown -R appuser:appuser /app/data/models

COPY --from=clustering-builder /opt/venv /opt/venv
COPY --chown=appuser:appuser --from=nlp-builder /opt/nltk_data /app/.cache/nltk_data

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
FROM runtime-base AS runtime-frontend

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

# =============================================================================
# Stage: runtime-rag
# RAG runtime with LangChain, OpenAI, and document embeddings
# Services: rag
# =============================================================================
FROM runtime-base AS runtime-rag

RUN mkdir -p /app/data/models /models/rag /models/hf-home && \
    chown -R appuser:appuser /app/data/models /models

COPY --from=rag-builder /opt/venv /opt/venv
COPY --chown=appuser:appuser --from=nlp-builder /opt/nltk_data /app/.cache/nltk_data

COPY --chown=appuser:appuser common/ /app/common/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser services/ /app/services/
COPY --chown=appuser:appuser scripts/ /app/scripts/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    RAG_MODELS_DIR=/models/rag \
    HF_HOME=/models/hf-home \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models \
    TORCH_HOME=/app/data/cache/torch \
    TRANSFORMERS_CACHE=/app/data/cache/transformers \
    NLTK_DATA=/app/.cache/nltk_data \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app
USER appuser
