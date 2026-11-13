# 🔍 Information Retrieval System

A production-ready, scalable Information Retrieval system built with microservices architecture, featuring multiple retrieval methods (TF-IDF, BM25, Embeddings, Hybrid) and a Streamlit interface.

## 🌟 Features

### Core IR Capabilities
- **Multiple Retrieval Methods**: TF-IDF, BM25, Word Embeddings (all-MiniLM-L6-v2)
- **Hybrid Search**: Serial and Parallel fusion strategies
- **Query Refinement**: Spell correction, synonym expansion
- **Performance Evaluation**: MAP, Precision@K, Recall, nDCG

### Production Features
- **Microservices Architecture**: 8 independent services with clear separation of concerns
- **Docker Optimization**: Multi-stage builds, layer caching, ~90% build time reduction
- **Health Monitoring**: Real-time metrics (CPU, memory, response times)
- **Database Optimization**: Indexed columns, materialized views for common queries
- **Resource Management**: CPU/Memory limits per service
- **Network Isolation**: Separate networks for frontend/backend/database layers

## 🏗️ Architecture

```
┌─────────────┐
│  Frontend   │ (Streamlit - Port 8501)
│  (500MB)    │
└──────┬──────┘
       │ frontend_net
       │
┌──────┴────────────────────────────────────────┐
│            API Gateway                        │
│         (FastAPI - Port 8000)                 │
│              (250MB)                          │
└───────────────┬───────────────────────────────┘
                │ backend_net
        ┌───────┴───────┐
        │               │
┌───────┴────┐  ┌──────┴────────┐  ┌──────────────┐
│Preprocessing│  │   Indexing    │  │  Retrieval   │
│   (NLP)     │  │     (ML)      │  │     (ML)     │
│   800MB     │  │    3.5GB      │  │    3.5GB     │
└─────────────┘  └───────────────┘  └──────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────┴────┐  ┌────────┴────┐  ┌───────┴──────┐
│  Ranking   │  │Query Refine │  │ Evaluation   │
│    (ML)    │  │    (NLP)    │  │    (ML)      │
│   3.5GB    │  │    800MB    │  │   3.5GB      │
└────────────┘  └─────────────┘  └──────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │    PostgreSQL    │
               │   (Data Store)   │
               └──────────────────┘
```

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

# Build all services (one command, no manual steps)
make build
```

**Build time**: ~30 minutes (first time with cold cache), ~7 seconds (incremental rebuild)

### 2. Start Database & Initialize

```bash
# Start PostgreSQL
docker compose up -d postgres

# Wait 10 seconds for PostgreSQL to be ready
sleep 10

# Initialize database schema (creates tables, indexes, materialized views)
make init-db
```

### 3. Data Preparation

#### Option A: If you have existing data in PostgreSQL

```bash
# Check what you have
make check-data

# Rebuild indices only
make build-indices
```

#### Option B: Starting from scratch

```bash
# Download dataset (using ir-datasets)
make download-dataset

# Preprocess data (NLP cleaning, tokenization)
make preprocess-data

# Load to database
make load-to-db

# Build all indices (TF-IDF, BM25, Embeddings)
make build-indices
```

**Time**: ~30-60 minutes depending on dataset size

### 4. Start All Services

```bash
# Start everything
make up

# Check health
make health
```

### 5. Access

- **Frontend UI**: http://localhost:8501
- **API Gateway**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Checks**: http://localhost:8000/health

## 📦 Project Structure

```
ir-project/
├── docker/                          # Dockerfiles
│   ├── python-base.Dockerfile      # Shared base (180MB)
│   ├── base.Dockerfile             # API services (250MB)
│   ├── service-nlp.Dockerfile      # NLP services (800MB)
│   ├── ml.Dockerfile               # ML services (3.5GB)
│   └── frontend.Dockerfile         # Streamlit (500MB)
│
├── requirements/                    # Split dependencies
│   ├── base.txt                    # Core APIs
│   ├── ml.txt                      # ML/DL libraries
│   ├── nlp.txt                     # NLP libraries
│   ├── ir.txt                      # IR-specific
│   └── frontend.txt                # Streamlit
│
├── services/                        # Microservices
│   ├── api_gateway/                # Entry point
│   ├── preprocessing/              # Text preprocessing
│   ├── indexing/                   # Index building
│   ├── retrieval/                  # Document retrieval
│   ├── ranking/                    # Result ranking
│   ├── query_refinement/           # Query enhancement
│   ├── evaluation/                 # Performance metrics
│   └── frontend/                   # Web UI
│
├── common/                          # Shared utilities
│   ├── database.py                 # PostgreSQL
│   ├── health.py                   # Health endpoints
│   ├── monitoring.py               # Metrics collection
│   └── logging_config.py           # Logging setup
│
├── scripts/                         # Management tools
│   ├── init_db.py                  # Initialize DB schema
│   ├── check_data.py               # Data status checker
│   ├── refresh_views.py            # Refresh materialized views
│   └── build_indices.py            # Build all indices
│
├── data/                            # Data volumes (Docker)
│   ├── datasets/                   # Raw datasets
│   ├── indices/                    # Built indices
│   ├── models/                     # ML models
│   ├── cache/                      # Temp cache
│   └── evaluation/                 # Results
│
├── docker-compose.yml               # Production config
├── docker-compose.dev.yml           # Development overrides
├── docker-compose.tools.yml         # Scripts & tools
├── Makefile                         # Automation commands
└── .env.example                     # Environment template
```

## 🛠️ Development

### Hot Reload Mode

```bash
# Start in development mode (code changes auto-reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Useful Commands

```bash
# View all available commands
make help

# View logs
make logs                # All services
make logs-api            # API Gateway only
make logs-retrieval      # Retrieval service

# Service management
make ps                  # Show running services
make restart             # Restart all
make health              # Health status

# Database
make refresh-views       # Refresh materialized views
make db-shell            # PostgreSQL shell
make db-backup           # Backup database

# Data checks
make check-data          # Status of datasets/indices

# Cleanup
make down                # Stop services
make down-volumes        # Stop + remove data
make clean-all           # Remove everything
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

## 📊 Performance Optimizations

### Build Time
- **Before**: 5 minutes per build
- **After**: 30 seconds (with cache)
- **Improvement**: 90% faster

### Image Sizes
| Service | Before | After | Reduction |
|---------|--------|-------|-----------|
| API Gateway | 2.1GB | 250MB | -88% |
| Preprocessing | 2.1GB | 800MB | -62% |
| Frontend | 2.1GB | 500MB | -76% |

### Runtime Performance
- **Startup Time**: 30 seconds (models preloaded)
- **Database Queries**: Optimized with indexes + materialized views

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

Each service has CPU/Memory limits:
- **API Gateway**: 1 CPU, 512MB RAM
- **Preprocessing**: 2 CPUs, 2GB RAM
- **ML Services**: 4 CPUs, 4GB RAM each

Adjust in `docker-compose.yml` if needed.

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
# Check resource usage
make stats

# Clean up
make down-volumes
docker system prune -a --volumes
```

## 💾 Disk Management

The Docker restructure achieves ~79% disk savings (39GB → 8GB after pruning).

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

### Current Disk Usage (after restructure)

- **Images**: 2.7GB (5 shared tier images)
- **Build Cache**: 5GB (cached pip wheels for incremental builds)
- **Volumes**: 539MB (models, datasets, indices)
- **Total**: ~8GB (vs. 39GB baseline, 79% reduction)

### BuildKit Cache Strategy

The unified Dockerfile uses BuildKit cache mounts for pip installs. This means:
- First build: ~30 minutes (downloads all dependencies)
- Incremental builds: ~7 seconds (reuses cached wheels)
- Adding a library: downloads only the new package

Cache is preserved across builds until you run `docker builder prune`.

## 📈 Monitoring

### Health Endpoints

Each service exposes:
- `/health` - Basic health check
- `/metrics` - Detailed metrics (CPU, memory, requests)
- `/readiness` - Readiness probe
- `/liveness` - Liveness probe

### Metrics Example

```bash
curl http://localhost:8000/metrics
```

```json
{
  "service": "api_gateway",
  "status": "healthy",
  "uptime_seconds": 145.23,
  "requests": {
    "total": 1234,
    "errors": 5,
    "error_rate": 0.004,
    "avg_response_time_ms": 23.4
  },
  "system": {
    "cpu_percent": 15.2,
    "memory_percent": 42.1,
    "process_memory_mb": 128.5
  }
}
```

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
- API docs: http://localhost:8000/docs
- Architecture diagrams: `/docs/architecture.md`
- Database schema: `/docs/database.md`
