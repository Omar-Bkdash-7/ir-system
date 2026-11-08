# System tests (whole pipeline, over HTTP)

These tests cross the unit boundaries: they call the **running** services and check that preprocessing, indexing,
retrieval, fusion and ranking agree with each other. They start nothing and stop nothing, so the stack has to be up:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d \
    preprocessing indexing retrieval ranking api_gateway
```

Then, from the project root (host Python, the dev ports are published):

```bash
python -m unittest discover -s tests/system -t . -v
```

Inside a container the URLs must point at the compose service names:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T \
    -e IR_GATEWAY_URL=http://api_gateway:8000 -e IR_PREPROCESSING_URL=http://preprocessing:8001 \
    -e IR_INDEXING_URL=http://indexing:8002 -e IR_RETRIEVAL_URL=http://retrieval:8003 \
    -e IR_RANKING_URL=http://ranking:8004 retrieval python -m unittest discover -s tests/system -t . -v
```

Every service that does not answer `/health` **skips** the suite with a message naming it, so the tests never fail
just because the stack is down.

## What they cover

| File | Needs | Checks |
|---|---|---|
| `test_pipeline.py` | an already indexed dataset (default `cranfield`) | preprocessing tokens, all three retrievers, the four fusion methods, determinism, gateway ↔ retrieval agreement, ranking on real fused output, ranking `/fuse` == retrieval fusion, and the error contracts (400/404/422 instead of 500) |
| `test_index_cycle.py` | a **downloaded** dataset, opt-in | upload to PostgreSQL → build → the built index is served → a custom `preprocessing_config` really changes the vocabulary → a refused rebuild leaves the index intact → cleanup |

Configuration (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `SYSTEM_TEST_DATASET` | `cranfield` | dataset with bm25 + tfidf + embedding indices already built |
| `SYSTEM_TEST_INDEX_DATASET` | unset | enables `test_index_cycle.py`; must already be downloaded (e.g. `beir/nfcorpus`) |
| `IR_*_URL` | `http://localhost:<port>` | service base URLs |

The lifecycle test writes a PostgreSQL table and an index directory, and deletes both in `tearDownModule` — also when a
test fails:

```bash
SYSTEM_TEST_INDEX_DATASET=beir/nfcorpus python -m unittest tests.system.test_index_cycle -v
```
