"""Persistence of evaluation results.

The only module here that knows SQL. Everything it returns is a plain dict in the
shape the API documents, so callers never touch cursors or column order.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from common.database import get_db_connection

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    model VARCHAR(100) NOT NULL,
    dataset VARCHAR(200) NOT NULL,
    top_k INTEGER NOT NULL,
    map_score FLOAT,
    ndcg_score FLOAT,
    recall_score FLOAT,
    precision_score FLOAT,
    num_queries INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluation_results(model);
CREATE INDEX IF NOT EXISTS idx_eval_dataset ON evaluation_results(dataset);
CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluation_results(created_at DESC);
"""

_SELECT_COLUMNS = """
    id, model, dataset, top_k, map_score, ndcg_score,
    recall_score, precision_score, num_queries, metadata, created_at
"""


@contextmanager
def _connection() -> Iterator[Any]:
    """One connection per operation, always closed."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - closing twice must not mask the real error
            pass


def _row_to_result(row) -> Dict[str, Any]:
    """Column order matches `_SELECT_COLUMNS`."""
    return {
        "id": row[0],
        "model": row[1],
        "dataset": row[2],
        "top_k": row[3],
        "map": row[4],
        "ndcg": row[5],
        "recall": row[6],
        "precision": row[7],
        "num_queries": row[8],
        "metadata": row[9],
        "created_at": row[10].isoformat() if row[10] else None,
    }


class ResultsStorage:
    """Store and retrieve evaluation results."""

    def __init__(self, *, ensure_table: bool = True) -> None:
        if ensure_table:
            self.ensure_table()

    def ensure_table(self) -> None:
        """Create the table and its indexes if they are missing."""
        try:
            with _connection() as conn:
                conn.cursor().execute(_CREATE_TABLE)
                conn.commit()
            logger.info("Evaluation results table verified")
        except Exception as exc:  # noqa: BLE001 - a missing database is reported, not fatal
            logger.error("Could not create the evaluation_results table: %s", exc, exc_info=True)

    # ------------------------------------------------------------------------------------------------- write
    def save_results(
        self,
        model: str,
        dataset: str,
        top_k: int,
        map_score: float,
        ndcg_score: float,
        recall_score: float,
        precision_score: float,
        num_queries: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Store one run, replacing the earlier run of the same model+dataset with the
        same refinement setting (a re-run of the same experiment is not a new row).
        """
        metadata = metadata or {}
        enable_refinement = bool(metadata.get("enable_refinement", False))
        metadata_json = json.dumps(metadata)

        with _connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM evaluation_results
                WHERE model = %s AND dataset = %s
                  AND (metadata->>'enable_refinement')::boolean = %s
                LIMIT 1;
                """,
                (model, dataset, enable_refinement),
            )
            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    """
                    UPDATE evaluation_results
                    SET top_k = %s, map_score = %s, ndcg_score = %s,
                        recall_score = %s, precision_score = %s, num_queries = %s,
                        metadata = %s, created_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id;
                    """,
                    (top_k, map_score, ndcg_score, recall_score, precision_score,
                     num_queries, metadata_json, existing[0]),
                )
                result_id = cursor.fetchone()[0]
                logger.info("Updated evaluation result id=%s (%s/%s)", result_id, model, dataset)
            else:
                cursor.execute(
                    """
                    INSERT INTO evaluation_results
                        (model, dataset, top_k, map_score, ndcg_score, recall_score,
                         precision_score, num_queries, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (model, dataset, top_k, map_score, ndcg_score, recall_score,
                     precision_score, num_queries, metadata_json),
                )
                result_id = cursor.fetchone()[0]
                logger.info("Stored evaluation result id=%s (%s/%s)", result_id, model, dataset)

            conn.commit()
            cursor.close()
            return result_id

    # ------------------------------------------------------------------------------------------------- read
    def get_latest_results(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Newest-first results, optionally filtered. A database error yields an empty list."""
        query = f"SELECT {_SELECT_COLUMNS} FROM evaluation_results WHERE 1=1"
        params: List[Any] = []

        if model:
            query += " AND model = %s"
            params.append(model)
        if dataset:
            query += " AND dataset = %s"
            params.append(dataset)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(max(1, int(limit)))

        try:
            with _connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                cursor.close()
            return [_row_to_result(row) for row in rows]
        except Exception as exc:  # noqa: BLE001 - the API degrades to "no history"
            logger.error("Could not read evaluation results: %s", exc, exc_info=True)
            return []

    def compare_models(self, dataset: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Average metrics per model for one dataset and cutoff, best MAP first."""
        try:
            with _connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT model,
                           AVG(map_score), AVG(ndcg_score),
                           AVG(recall_score), AVG(precision_score),
                           COUNT(*)
                    FROM evaluation_results
                    WHERE dataset = %s AND top_k = %s
                    GROUP BY model
                    ORDER BY AVG(map_score) DESC;
                    """,
                    (dataset, top_k),
                )
                rows = cursor.fetchall()
                cursor.close()
            return [
                {
                    "model": row[0],
                    "avg_map": float(row[1] or 0.0),
                    "avg_ndcg": float(row[2] or 0.0),
                    "avg_recall": float(row[3] or 0.0),
                    "avg_precision": float(row[4] or 0.0),
                    "num_evaluations": row[5],
                }
                for row in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not compare models for dataset %s: %s", dataset, exc, exc_info=True)
            return []

    # ------------------------------------------------------------------------------------------------- delete
    def delete_result(self, result_id: int) -> bool:
        """Delete one stored run; False when it does not exist."""
        try:
            with _connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM evaluation_results WHERE id = %s RETURNING id;", (result_id,))
                deleted = cursor.fetchone()
                conn.commit()
                cursor.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not delete evaluation result %s: %s", result_id, exc, exc_info=True)
            return False

        if deleted:
            logger.info("Deleted evaluation result id=%s", result_id)
            return True
        logger.warning("Evaluation result id=%s not found", result_id)
        return False

    def delete_by_filters(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        before_date: Optional[str] = None,
    ) -> int:
        """
        Delete every run matching the filters.

        Refuses to run without a filter: an accidental "delete everything" from a
        mis-built request is worse than a 400.
        """
        if not any((model, dataset, before_date)):
            raise ValueError("At least one filter (model, dataset, before_date) is required")

        query = "DELETE FROM evaluation_results WHERE 1=1"
        params: List[Any] = []

        if model:
            query += " AND model = %s"
            params.append(model)
        if dataset:
            query += " AND dataset = %s"
            params.append(dataset)
        if before_date:
            query += " AND created_at < %s"
            params.append(before_date)

        query += " RETURNING id;"

        try:
            with _connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                count = len(cursor.fetchall())
                conn.commit()
                cursor.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not delete evaluation results: %s", exc, exc_info=True)
            return 0

        logger.info("Deleted %d evaluation results (model=%s, dataset=%s, before=%s)",
                    count, model, dataset, before_date)
        return count

    def clear_all(self) -> int:
        """Delete every stored run (used by the maintenance endpoint only)."""
        try:
            with _connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM evaluation_results RETURNING id;")
                count = len(cursor.fetchall())
                conn.commit()
                cursor.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not clear evaluation results: %s", exc, exc_info=True)
            return 0

        logger.warning("Cleared all evaluation results (%d rows)", count)
        return count
