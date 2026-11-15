# Scripts

One-shot jobs. Every script here is wrapped by `docker-compose.tools.yml` (or the
CI stack), so they run inside the same images the services use — nothing needs to
be installed on the host.

Run them from the repository root:

```bash
TOOLS="docker compose -f docker-compose.yml -f docker-compose.tools.yml --profile tools"
$TOOLS run --rm -T <tool-name>
```

## Database tools

| Script | Tool | Purpose |
|---|---|---|
| `init_db.py` | `init_db` | Create the schema: tables, indexes, materialized views. Idempotent. |
| `refresh_views.py` | `refresh_views` | Refresh the materialized views after documents changed. |
| `check_data.py` | `check_data` | Status report: datasets on disk, indices, database rows. |

## Data pipeline

Run in order to go from raw data to searchable indices:

| Script | Tool | Purpose |
|---|---|---|
| `download_dataset.py` | `download_dataset` | Fetch a dataset with `ir-datasets` into `data/datasets/.ir_datasets`. |
| `preprocess_dataset.py` | `preprocess_dataset` | NLP cleaning and tokenization into `data/processed`. |
| `load_documents_to_db.py` | `load_to_db` | Write the processed documents into PostgreSQL. |
| `build_indices.py` | `build_indices` | Build TF-IDF + BM25 + embedding indices in one go. |

Single-index variants for rebuilds:

| Script | Tool | Purpose |
|---|---|---|
| `build_tfidf_index.py` | `build_tfidf` | TF-IDF index only. |
| `build_bm25_index.py` | `build_bm25` | BM25 index only. |
| `build_embedding_index.py` | `build_embeddings` | Embedding index only (loads the embedding model). |

## Evaluation and CI

| Script | Purpose |
|---|---|
| `evaluate_retrieval.py` (`run_evaluation`) | Score a model over a dataset: MAP, P@K, nDCG; writes `data/evaluation/`. |
| `ci_seed.py` | Seeds the three-document contract-test dataset used by `docker-compose.ci.yml` (`seed` service). |
| `validate_clustering.py` | Structural check of the clustering service: imports resolve, package layout complete, service wired into configuration. Run it inside a container, e.g. `docker run --rm -v "$PWD":/repo -w /repo ir-clustering:latest python scripts/validate_clustering.py`. |

## Conventions

- `pytest.ini` excludes `scripts/` from collection — these are jobs, not tests. Real
  tests live in `tests/`.
- The tools carry the `tools` profile, so `docker compose up` never starts them by
  accident; they appear only when invoked through the `--profile tools` commands above.
