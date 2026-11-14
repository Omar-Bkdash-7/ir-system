# 🔍 Information Retrieval System

An Information Retrieval system built as microservices: eight FastAPI services behind one gateway, a Streamlit interface, and multiple retrieval methods (TF-IDF, BM25, Embeddings, Hybrid).

**Status:** the services and the three test levels (unit, contract, lint) are in place and passing. Two things are deliberately not there yet, and the rest of this file does not pretend otherwise: compose runs the development overrides only (a production variant is described but not written), and there is no metrics layer — observability is `/health`, container healthchecks and rotating log files (see [Observability](#-observability)). The images are built from the same requirements the tests use, so a rebuild is all that is needed for an existing image to carry `pytest` and `ruff`.

## 🌟 Features

### Core IR Capabilities
- **Multiple Retrieval Methods**: TF-IDF, BM25, Word Embeddings (all-MiniLM-L6-v2)
- **Hybrid Search**: Serial and Parallel fusion strategies
- **Query Refinement**: Spell correction, synonym expansion
- **Performance Evaluation**: MAP, Precision@K, Recall, nDCG

### Engineering Features
- **Microservices**: one gateway plus eight services (preprocessing, indexing, retrieval, ranking, query_refinement, evaluation, clustering, rag), and a Streamlit frontend that talks to the gateway only
- **Shared image tiers**: six images, one per dependency tier, so adding a library rebuilds one tier instead of every service (sizes below)
- **Database**: indexed columns and materialized views for the common queries
- **Resource limits**: `cpus` / `mem_limit` / `mem_reservation` on all twelve blocks in `docker-compose.yml`
- **Network isolation**: three networks (frontend / backend / database)
- **Logging**: one logger per service, console plus a rotating file handler (10MB × 5) under `logs/`

## 🏗️ Architecture

```
frontend (Streamlit, 8501)
  │ frontend_net
api_gateway (FastAPI, 8000)        CORS, rate limit, timeouts, retries
  │ backend_net
  ├── preprocessing   8001   nlp tier
  ├── indexing        8002   ml tier
  ├── retrieval       8003   ml tier    (owns the indices and the embedding model)
  ├── ranking         8004   light tier
  ├── query_refinement 8005  ml tier
  ├── evaluation      8006   ml tier
  ├── clustering      8007   clustering tier
  └── rag             8008   rag tier
        │ database_net
     postgres:15     5432
```

The names on the right are the image tiers from `docker-compose.yml`; their sizes are in [Image sizes](#image-sizes). The frontend talks to the gateway only — never to a service directly.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 16GB RAM minimum (20GB+ recommended)
- 50GB disk space

### 1. Initial Setup (First Time Only)

```bash
# Clone repository
git clone <repo-url>
cd ir-project

# Copy environment file
cp .env.example .env

# Edit .env if needed (passwords, ports, etc.)
nano .env

# Build the six tier images (one command, no manual steps)
docker compose build
```

**Build time**: ~30 minutes on a cold cache (network-bound), a few seconds when nothing changed.

### 2. Start Database & Initialize

```bash
# Start PostgreSQL and wait for its healthcheck instead of guessing a delay
docker compose up -d --wait postgres

# Initialize database schema (creates tables, indexes, materialized views)
docker compose -f docker-compose.yml -f docker-compose.tools.yml --profile tools run --rm -T init_db
```

### 3. Data Preparation

The one-off data jobs live in `docker-compose.tools.yml`: each is a container that runs a single script and exits. `--profile tools` is what makes compose accept them.

```bash
TOOLS="docker compose -f docker-compose.yml -f docker-compose.tools.yml --profile tools"

# Option A - you already have documents and indices
$TOOLS run --rm -T check_data      # status of the datasets, indices and database
$TOOLS run --rm -T build_indices   # rebuild TF-IDF + BM25 + embeddings

# Option B - start from raw data
$TOOLS run --rm -T download_dataset    # fetch a dataset with ir-datasets
$TOOLS run --rm -T preprocess_dataset  # NLP cleaning and tokenization
$TOOLS run --rm -T load_to_db          # write documents to PostgreSQL
$TOOLS run --rm -T build_indices       # build TF-IDF, BM25 and embedding indices
```

Single-index variants exist as `build_tfidf`, `build_bm25` and `build_embeddings`; `run_evaluation` scores a model over a dataset with `scripts/evaluate_retrieval.py`.

**Time**: ~30-60 minutes depending on dataset size

### 4. Start All Services

```bash
# Development stack: bind mounts and --reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Every service answers /health
for p in 8000 8001 8002 8003 8004 8005 8006 8007 8008; do
  printf '%s ' "$p"; curl -fsS -o /dev/null -w '%{http_code}\n' "http://localhost:$p/health"
done
```

### 5. Access

- **Frontend UI**: http://localhost:8501
- **API Gateway**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Checks**: http://localhost:8000/health

## 📦 Project Structure

```
ir/
├── docker/
│   └── unified.Dockerfile          # Every stage: system → base-builder → ir/scientific/nlp/
│                                   #   torch/transformers/clustering/rag builders → runtime-* targets
│
├── requirements/                   # Dependency tiers, installed in order by the Dockerfile
│   ├── 00-system.txt               # pip/setuptools floor (no OS packages here)
│   ├── 01-base.txt                 # fastapi, uvicorn, httpx, pydantic, numpy, pandas, psycopg2
│   ├── 02-scientific.txt           # scikit-learn, scipy (builds on 01)
│   ├── 03-nlp.txt                  # spacy, nltk, without torch
│   ├── 04-torch.txt                # CPU-only torch; the version is held to avoid CUDA wheels
│   ├── 05-transformers.txt         # transformers + sentence-transformers (builds on 04)
│   ├── 06-clustering.txt           # hdbscan, umap-learn (topics come from c-TF-IDF, not BERTopic)
│   ├── 07-ir.txt                   # ir-datasets, rank-bm25, ir-measures
│   ├── 08-frontend.txt             # streamlit, plotly
│   ├── 09-rag.txt                  # HuggingFace Inference API client; no local models
│   ├── 10-dev.txt                  # pytest + ruff (every image, for the suite and the lint gate)
│   ├── constraints.txt             # Pinned versions
│   └── requirements.in             # pip-compile input
│
├── services/                       # Microservices
│   ├── api_gateway/                # Entry point, CORS + rate limit, calls the others
│   ├── preprocessing/              # Text preprocessing
│   ├── indexing/                   # Index building
│   ├── retrieval/                  # Document retrieval
│   ├── ranking/                    # Result ranking
│   ├── query_refinement/           # Query enhancement
│   ├── evaluation/                 # Performance metrics
│   ├── clustering/                 # Topic clustering
│   └── rag/                        # Retrieval-augmented generation
│
├── frontend/                       # Streamlit app (app.py, app_pages/, components/, api/, styles/)
│
├── common/                         # Shared, imported by every service
│   ├── config.py                   # YAML + environment settings
│   ├── database.py                 # PostgreSQL
│   ├── security.py                 # Unified CORS + rate limiting
│   ├── logging_config.py           # Per-service logger, rotating file handler
│   ├── http_client.py              # Inter-service calls with retries/timeouts
│   ├── exceptions.py               # Also: exception_handlers.py
│   ├── dataset_context.py          # Active-dataset resolution
│   ├── device.py                   # CPU/CUDA selection
│   └── model_loader.py             # Cached model loading
│
├── configs/config.yaml             # Defaults every service reads through common/config.py
├── scripts/                        # One-shot jobs, each wrapped by docker-compose.tools.yml
│   ├── init_db.py                  # Schema, indexes, materialized views
│   ├── check_data.py               # Data/indices status
│   ├── refresh_views.py            # Refresh materialized views
│   ├── build_indices.py            # All indices (also build_tfidf/bm25/embedding_index.py)
│   ├── download_dataset.py, preprocess_dataset.py, load_documents_to_db.py
│   ├── evaluate_retrieval.py       # Evaluation runs
│   └── ci_seed.py                  # Seeds the contract-test dataset
│
├── tests/                          # pytest suites, one package per service plus frontend/, integration/, system/
├── data/                           # Bind-mounted in dev: datasets, indices, models, cache,
│                                   #   processed, evaluation, metadata, clusters
├── logs/                           # Rotating per-service logs
├── docs/                           # Design notes, deployment guide, user guide
│
├── docker-compose.yml              # Base stack (all services, limits, healthchecks)
├── docker-compose.dev.yml          # Development overrides: bind mounts, --reload
├── docker-compose.tools.yml        # One-shot scripts (init_db, build_indices, ...)
├── docker-compose.ci.yml           # Contract-test stack: postgres + 3 services + seed + tests
├── pytest.ini                      # Collection limited to tests/
├── ruff.toml                       # Lint rules (pyflakes only)
└── .env.example                    # Environment template
```

## 🛠️ Development

### Hot Reload Mode

```bash
# Start in development mode (code changes auto-reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Useful Commands

```bash
DEV="docker compose -f docker-compose.yml -f docker-compose.dev.yml"
TOOLS="docker compose -f docker-compose.yml -f docker-compose.tools.yml --profile tools"

# Logs
$DEV logs -f                 # all services
$DEV logs -f api_gateway     # one service
$DEV logs -f retrieval

# Service management
$DEV ps                      # running services and their health
$DEV restart                 # restart all
$DEV restart api_gateway     # restart one

# Database
$TOOLS run --rm -T refresh_views        # refresh materialized views
$DEV exec postgres psql -U ${POSTGRES_USER:-ir_user} -d ${POSTGRES_DB:-ir_db}
$DEV exec -T postgres pg_dump -U ${POSTGRES_USER:-ir_user} ${POSTGRES_DB:-ir_db} > backup.sql

# Data
$TOOLS run --rm -T check_data           # status of datasets/indices

# Cleanup
$DEV down                    # stop services
$DEV down -v                 # stop + remove the data volumes
docker system prune -a --volumes   # remove everything: images, cache, volumes
```

### Tests

The suite runs with pytest, which is installed in every image through `requirements/10-dev.txt` (no ad-hoc install needed after a rebuild):

```bash
# Whole suite inside the dev stack
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T query_refinement python -m pytest tests

# One module only
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T query_refinement python -m pytest tests/query_refinement
```

`pytest.ini` limits collection to `tests/`, so standalone helper scripts under `scripts/` are not picked up.

`tests/frontend` exercises the Streamlit app without a browser: retry/error normalization in `api/client.py`, the request payloads of the service clients, and the stylesheet contract (every `inject_style("name")` must resolve to `frontend/styles/name.css`). It imports `frontend/`, which the dev compose mounts into this service.

#### Contract test (real services)

Unit tests fake their neighbours, so a renamed field or a changed status code between two services stays invisible. `docker-compose.ci.yml` runs the smallest stack that can disagree with itself (postgres, preprocessing, retrieval, api_gateway), seeds one three-document dataset and asserts the contract the frontend depends on:

```bash
# Rebuild the images first when service code changed: the stack runs the code baked into them.
docker compose -f docker-compose.ci.yml up -d --wait postgres preprocessing retrieval api_gateway
docker compose -f docker-compose.ci.yml run --rm seed
docker compose -f docker-compose.ci.yml run --rm tests
docker compose -f docker-compose.ci.yml down -v
```

The same sequence runs in the `contract` job of `.github/workflows/tests.yml`, which builds the three images itself. No host ports are published, so it can run next to a dev stack.

#### Lint

`ruff` comes from the same dev requirements, so the gate that runs in CI can be run against the checkout without installing anything on the host. The rules live in `ruff.toml` at the repository root and ruff must see that file, so the whole checkout is mounted at `/repo` — the dev compose mounts only `services/`, `common/` and `tests/`, which would hide `frontend/`, `scripts/` and the config:

```bash
# Any image built after ruff was added to requirements/10-dev.txt carries it.
docker run --rm -v "$PWD":/repo -w /repo ir-light:latest ruff check .

# Windows Git Bash needs the path made literal, otherwise it rewrites /repo:
MSYS_NO_PATHCONV=1 docker run --rm -v "D:/path/to/ir:/repo" -w /repo ir-light:latest ruff check .
```

Only the pyflakes rules (F) are enabled: unused imports and variables, undefined names, redefinitions. The test suite covers the branches it enters; this covers the rest — the `undefined name` class of bug only fires on the path nobody exercised. The same command is the `lint` job in `.github/workflows/tests.yml`, pinned to the ruff version in `requirements/constraints.txt`.

## 📊 Image sizes and build time

<a id="image-sizes"></a>

Measured on this checkout with `docker images`; one image serves every service in its tier:

| Image | Services | Size |
|---|---|---|
| `ir-light:latest` | api_gateway, ranking | 615MB |
| `ir-frontend:latest` | frontend | 765MB |
| `ir-nlp:latest` | preprocessing | 835MB |
| `ir-ml:latest` | indexing, retrieval, query_refinement, evaluation | 1.84GB |
| `ir-rag:latest` | rag | 1.9GB |
| `ir-clustering:latest` | clustering | 2.23GB |

The spread is the dependency tiers, not waste: CPU torch and transformers dominate `ir-ml`, and `ir-clustering` adds hdbscan/UMAP on top of it. `ir-rag` is large mostly because it inherits the same chain.

How the sizes translate into build time:

- **First build**: ~30 minutes, almost entirely downloading dependencies.
- **Rebuild with the same requirements**: a few seconds — BuildKit cache mounts keep the pip wheels.
- **Adding a library**: only the tier that needs it rebuilds, and only the new wheel downloads.

Runtime: services start in seconds. The first call that needs an embedding model pays for loading it, then it stays in `data/models` and the process keeps it in memory — the retrieval container's memory footprint after loading the `vaswani` index was measured at ~117MB, which is why it runs two uvicorn workers (`RETRIEVAL_UVICORN_WORKERS`).

## 🔧 Configuration

### Rate limiting and CORS

`common/security.py` installs the CORS policy in every service and the request limit in the API gateway.
`RATE_LIMIT` is the number of requests one client address may make per `RATE_LIMIT_WINDOW` seconds (`0` disables it); `/health`, `/metrics`, `/docs` and the other probe routes stay exempt, so health checks never trip it. Both are installed last, so a throttled `429` still carries `Retry-After` and the CORS headers.

### Environment Variables

Key variables in `.env`:

```bash
# Database
POSTGRES_USER=ir_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=ir_db

# Service Ports
API_GATEWAY_PORT=8000
FRONTEND_PORT=8501

# Security
SECRET_KEY=generate_with_openssl_rand_hex_32
CORS_ORIGINS=http://localhost:8501
```

### Service Tuning (optional)

Each service reads its own defaults; these variables only change them. See the docstring at the top of
each module for the full table and the matching `configs/config.yaml` key.

| Variable | Default | Effect |
| --- | --- | --- |
| `EVALUATION_WORKERS` | `8` | Queries evaluated in parallel (1 = serial, capped at 64). `retrieval` must be able to answer concurrently for this to pay off. |
| `RETRIEVAL_UVICORN_WORKERS` | `1` | Worker processes of the retrieval container. Each worker loads its own copy of the indices: raise it only when memory allows. |
| `DB_POOL_ENABLED` | `true` | Reuse PostgreSQL connections instead of opening one per query. Set `false` to get the old behaviour. |
| `DB_POOL_MIN` / `DB_POOL_MAX` | `1` / `8` | Bounds of that pool. |
| `GATEWAY_RETRIEVAL_TIMEOUT` | `30` | Seconds the API gateway waits for a retrieval answer. |
| `GATEWAY_HYBRID_TIMEOUT` | `60` | Same, for the two hybrid search paths (they run several models). |
| `GATEWAY_REFINEMENT_TIMEOUT` | `10` | Same, for query refinement. |
| `GATEWAY_RAG_TIMEOUT` | `300` | Same, for RAG generation — CPU generation is slow on purpose. |
| `GATEWAY_MAX_RETRIES` / `GATEWAY_RETRY_BACKOFF` | `2` / `0.2` | Retries on a 5xx or a refused connection, and the delay between them. |
| `GATEWAY_PREPROCESSING_URL`, `GATEWAY_RETRIEVAL_URL`, `GATEWAY_QUERY_REFINEMENT_URL`, `GATEWAY_RAG_URL` | compose service names | Override only when the gateway runs outside compose. |

A value that cannot be parsed is logged and ignored, never fatal.

### Resource Limits

Set with `cpus` / `mem_limit` / `mem_reservation` (the keys compose actually enforces outside swarm; `deploy.resources` is ignored there):

| Service | cpus | mem_limit |
|---|---|---|
| postgres | 2.0 | 2G |
| api_gateway | 1.0 | 1G |
| frontend | 1.0 | 1G |
| preprocessing | 2.0 | 2G |
| ranking | 2.0 | 2G |
| query_refinement | 2.0 | 2G |
| evaluation | 2.0 | 2G |
| rag | 2.0 | 4G |
| indexing | 4.0 | 4G |
| retrieval | 4.0 | 4G |
| clustering | 4.0 | 4G |

## 🐛 Troubleshooting

### Service won't start (unhealthy)

```bash
# Check logs
docker compose logs <service_name>

# Example
docker compose logs preprocessing
```

### Database connection failed

```bash
# Check PostgreSQL
docker compose ps postgres

# Should show (healthy)
# If not, restart:
docker compose restart postgres
```

### Out of memory

```bash
# Check resource usage against the configured limits
docker stats --no-stream

# Clean up
$DEV down -v
docker system prune -a --volumes
```

## 💾 Disk Management

Images, build cache and data volumes are all on the same disk; measure before trusting any figure, because the numbers move with the datasets and models that have been downloaded.

### Check Disk Usage

```bash
# Overview
docker system df

# Detailed breakdown
docker system df -v
```

### Recommended Maintenance

```bash
# Prune old images (safe - only removes unused images)
docker image prune -a -f

# Prune build cache, keep 5GB of recent cache for fast rebuilds
docker builder prune --keep-storage=5gb

# Full cleanup (WARNING: next build will re-download everything)
docker system prune -a -f
docker builder prune -a -f
```

### Where the space goes

```bash
docker system df -v
```

- **Images**: the sum of the six tiers in [Image sizes](#image-sizes), ~8GB.
- **Build cache**: pip wheels kept by the BuildKit cache mounts; this is what makes incremental builds take seconds.
- **Volumes and bind mounts**: `data/` (datasets, indices, models) grows the most — embedding models alone are several hundred MB per model.

### BuildKit Cache Strategy

The unified Dockerfile uses BuildKit cache mounts for pip installs. This means:
- First build: ~30 minutes (downloads all dependencies)
- Incremental builds: ~7 seconds (reuses cached wheels)
- Adding a library: downloads only the new package

Cache is preserved across builds until you run `docker builder prune`.

## 📈 Observability

### Health

Every service exposes `GET /health`, which is a liveness check: the process answers and its imports resolved. Compose uses the same route for the container healthchecks (30s interval, 10s timeout, 3 retries, 40s start period).

```bash
curl -fsS http://localhost:8000/health
```

There are no `/readiness` or `/liveness` routes: those paths appear in `common/security.py` only so that, if a probe is added later, it is exempt from the rate limit.

### Metrics

The application itself does not sample CPU or memory and does not expose a Prometheus endpoint. For resource usage use Docker:

```bash
docker stats --no-stream
docker compose ps
```

The only `/metrics` route in the system belongs to the evaluation service (`GET /metrics`, plus `/metrics/per-query` and `/metrics/refinement-comparison`), and it reports retrieval quality per model — MAP, P@K, nDCG — not system resources.

### Logs

`common/logging_config.py` gives each service a console handler and a rotating file handler at `logs/<service>.log` (10MB per file, 5 backups), reusing the `logs` volume. Set `LOG_LEVEL` (default `INFO`) to change verbosity; if the file cannot be created the service continues with console logging only.

## 🔒 Security

- ✅ Non-root users in all containers
- ✅ Network isolation (3 separate networks)
- ✅ Resource limits prevent DoS
- ✅ Secrets via environment variables
- ✅ Minimal base images (python:3.11-slim)
- ✅ Health checks for all services
- ✅ Centralized logging with rotation

## 📝 License

[Your License Here]

## 👥 Contributors

[Your Name/Team]

## 📚 Documentation

For detailed documentation, see:
- API docs: http://localhost:8000/docs (generated by FastAPI from the gateway's routes)
- `docs/ir.md` — design notes for the retrieval side
- `docs/run.md` — running the stack step by step
- `docs/deployment-guide.md` — deployment notes
- `docs/RAG.md` — the RAG service
- `docs/project-context.md` — how the pieces fit together
- `docs/user-guide/` — end-user guide (Arabic and English)
