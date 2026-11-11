"""PostgreSQL access shared by every service.

A connection used to be opened on every call, so an evaluation run or a dataset
uploaded row-by-row paid one TCP handshake and one PostgreSQL backend process each
time. `get_db_connection()` now hands out pooled connections while keeping its old
contract: callers still call `close()`, which now returns the connection to the
pool instead of tearing it down. No call site needed to change.

    DB_POOL_ENABLED   true   set to false for one connection per call (old behaviour)
    DB_POOL_MIN       1      connections opened when the pool is first used
    DB_POOL_MAX       8      upper bound; a caller beyond it waits, then creates nothing
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import psycopg2
import psycopg2.extensions
from psycopg2 import pool as pg_pool
from psycopg2.extras import execute_batch

from common.config import get_config

logger = logging.getLogger(__name__)

_pool: Optional[pg_pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _dsn() -> str:
    """DATABASE_URL when it is set (containers), otherwise the database.* configuration."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    return psycopg2.extensions.make_dsn(
        host=get_config("database.host", "postgres"),
        port=get_config("database.port", 5432),
        user=get_config("database.user", "ir_user"),
        password=get_config("database.password", "ir_pass"),
        dbname=get_config("database.database", "ir_db"),
    )


def _pool_enabled() -> bool:
    return str(os.getenv("DB_POOL_ENABLED", "true")).strip().lower() not in ("0", "false", "no", "off")


def _pool_bounds() -> tuple[int, int]:
    try:
        minimum = max(1, int(os.getenv("DB_POOL_MIN", "1")))
    except ValueError:
        minimum = 1
    try:
        maximum = max(minimum, int(os.getenv("DB_POOL_MAX", "8")))
    except ValueError:
        maximum = 8
    return minimum, maximum


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            minimum, maximum = _pool_bounds()
            _pool = pg_pool.ThreadedConnectionPool(minimum, maximum, _dsn())
            logger.info("Database connection pool ready (min=%d, max=%d)", minimum, maximum)
    return _pool


def close_pool() -> None:
    """Close every pooled connection (shutdown, or a test that must start clean)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


class PooledConnection:
    """A psycopg2 connection whose ``close()`` returns it to the pool.

    Everything else — ``cursor()``, ``commit()``, ``rollback()``, ``closed``,
    ``with conn:`` — is delegated to the real connection, which is why the existing
    call sites keep working untouched.
    """

    __slots__ = ("_connection", "_pool", "_returned")

    def __init__(self, connection: Any, pool: pg_pool.ThreadedConnectionPool) -> None:
        self._connection = connection
        self._pool = pool
        self._returned = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def __enter__(self) -> "PooledConnection":
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> Any:
        return self._connection.__exit__(exc_type, exc, traceback)

    def close(self) -> None:
        if self._returned:
            return
        self._returned = True
        _release(self._connection, self._pool)

    # Some callers pass the connection around and check it directly.
    @property
    def closed(self) -> int:
        return self._connection.closed


def _release(connection: Any, pool: pg_pool.ThreadedConnectionPool) -> None:
    """Return a connection to the pool, ending anything still open on it first."""
    if connection.closed:
        pool.putconn(connection, close=True)
        return

    try:
        # Never hand a pool member back with an open or aborted transaction.
        if connection.info.transaction_status != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
            connection.rollback()
    except psycopg2.Error as exc:
        logger.debug("Discarding a connection that could not be reset: %s", exc)
        pool.putconn(connection, close=True)
        return

    try:
        pool.putconn(connection)
    except psycopg2.Error as exc:  # a full pool refuses the connection; drop it instead
        logger.debug("Closing a connection the pool refused: %s", exc)
        connection.close()


def get_connection():
    """Legacy: a brand new connection. Prefer get_db_connection()."""
    return psycopg2.connect(_dsn())


def get_db_connection():
    """
    A database connection. ``close()`` returns it to the shared pool.

    Falls back to a fresh connection when pooling is disabled, and raises
    ``psycopg2.OperationalError`` when the database is unreachable — callers already
    treat that as "database unavailable".
    """
    if not _pool_enabled():
        return psycopg2.connect(_dsn())
    return PooledConnection(_get_pool().getconn(), _get_pool())


def execute_query(conn, query: str, params=None):
    """
    Execute a SQL query (INSERT, UPDATE, DELETE, CREATE, etc.).
    Does not return results - use for non-SELECT queries.
    """
    cursor = conn.cursor()
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Query execution failed: {e}")
        raise
    finally:
        cursor.close()


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    table_name = get_config("database.table", "documents")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            doc_id VARCHAR(255) PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)

    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_doc_id ON {table_name}(doc_id)")

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Database table '{table_name}' initialized")


def insert_documents(documents: list[dict]):
    conn = get_db_connection()
    cur = conn.cursor()

    table_name = get_config("database.table", "documents")

    data = [(doc["doc_id"], doc["text"]) for doc in documents]

    execute_batch(
        cur,
        f"INSERT INTO {table_name} (doc_id, text) VALUES (%s, %s) ON CONFLICT (doc_id) DO NOTHING",
        data,
        page_size=1000
    )

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Inserted {len(documents)} documents into database")


def get_documents_by_ids(doc_ids: list[str], dataset: str = None) -> dict:
    conn = get_db_connection()
    cur = conn.cursor()

    # Use dataset-specific table if dataset is provided
    if dataset:
        safe_dataset = dataset.replace("/", "_").replace("-", "_")
        table_name = f"dataset_{safe_dataset}"
    else:
        table_name = get_config("database.table", "documents")

    cur.execute(
        f"SELECT doc_id, text FROM {table_name} WHERE doc_id = ANY(%s)",
        (doc_ids,)
    )

    results = {row[0]: row[1] for row in cur.fetchall()}

    cur.close()
    conn.close()

    return results
