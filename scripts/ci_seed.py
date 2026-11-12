"""Seed the contract-test stack with one tiny, fully deterministic dataset.

Runs inside the ML image (PostgreSQL client, scikit-learn and the indexers are
there) and writes exactly what the live services read back:

    postgres   dataset_ci_demo: three documents
    indices    data/indices/ci_demo/tfidf and /bm25
    metadata   data/metadata/ci-demo.json (the file the gateway reads for /datasets/info)

The documents are tokenised with a preprocessing config that turns off stemming and
stop-word removal, and that config is stored with the metadata: the gateway sends it
back to the preprocessing service for each query, so both sides tokenise the same way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATASET = "ci/demo"
TABLE = "dataset_ci_demo"
PREPROCESSING_CONFIG = {"use_stemming": False, "remove_stopwords": False}

DOCUMENTS = [
    ("d1", "information retrieval models rank documents by relevance"),
    ("d2", "the bm25 model scores documents with term frequencies"),
    ("d3", "vector space models compare query and document weights"),
]


def seed_database() -> None:
    from common.database import get_db_connection

    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} (doc_id VARCHAR(255) PRIMARY KEY, text TEXT NOT NULL)"
        )
        cursor.executemany(
            f"INSERT INTO {TABLE} (doc_id, text) VALUES (%s, %s) "
            "ON CONFLICT (doc_id) DO UPDATE SET text = EXCLUDED.text",
            DOCUMENTS,
        )
        connection.commit()
        cursor.close()
    finally:
        connection.close()
    print(f"seeded {len(DOCUMENTS)} documents into {TABLE}")


def processed_file() -> Path:
    path = Path("data/processed/ci_demo.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc_id, text in DOCUMENTS:
            handle.write(json.dumps({"doc_id": doc_id, "text": text, "tokens": text.split()}) + "\n")
    return path


def seed_indices() -> None:
    from services.indexing.indexers import BM25Indexer, TFIDFIndexer

    path = processed_file()
    for indexer_class in (TFIDFIndexer, BM25Indexer):
        indexer = indexer_class(dataset_name=DATASET)
        indexer.build(processed_path=str(path))
        indexer.save()
        print(f"built {indexer_class.__name__} for {DATASET}")


def seed_metadata() -> None:
    from services.indexing.dataset.metadata_manager import DatasetMetadataManager

    DatasetMetadataManager().save_metadata(
        {
            "dataset_name": DATASET,
            "doc_count": len(DOCUMENTS),
            "query_count": 0,
            "qrels_count": 0,
            "validation_status": "success",
            "preprocessing_config": PREPROCESSING_CONFIG,
        },
        dataset_name=DATASET,
    )
    print(f"wrote metadata for {DATASET}")


def main() -> int:
    try:
        seed_database()
        seed_indices()
        seed_metadata()
    except Exception as exc:  # noqa: BLE001 - a seed failure must fail the CI step loudly
        print(f"SEED FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("seed complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
