"""Document texts from PostgreSQL through a small connection pool.

The former code opened a new database connection for every request. Here a pool is created on first use and shared.
The table name follows the indexing service's rule (prefix + dataset slug) and is validated and quoted as an identifier.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Callable, Dict, Optional, Sequence

from common.config import get_config
from common.exceptions import InvalidQueryError

logger = logging.getLogger(__name__)

_SAFE_SLUG = re.compile(r"^[a-z0-9_]+$")


def table_name(dataset: str, prefix: Optional[str] = None) -> str:
    """`<prefix>_<slug>` exactly as services.indexing.naming.table_name builds it ('/', '-', '.' -> '_', lower case)."""
    slug = dataset.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    if not _SAFE_SLUG.match(slug):
        raise InvalidQueryError(f"Dataset id {dataset!r} cannot be used as a database table name")
    return f"{prefix or get_config('database.table_prefix', 'dataset')}_{slug}"


def _connection_arguments() -> Dict[str, Any]:
    """Same source as common.database.get_db_connection: DATABASE_URL, otherwise the database.* configuration."""
    url = os.getenv("DATABASE_URL")
    if url:
        return {"dsn": url}
    return {
        "host": get_config("database.host", "postgres"),
        "port": get_config("database.port", 5432),
        "user": get_config("database.user", "ir_user"),
        "password": get_config("database.password", "ir_pass"),
        "database": get_config("database.database", "ir_db"),
    }


class TextStore:
    def __init__(self, pool_max: int = 8, *, fetcher: Optional[Callable[[str, Sequence[str]], Dict[str, str]]] = None) -> None:
        self._pool_max = pool_max
        self._fetcher = fetcher  # injected in tests; None = PostgreSQL
        self._pool: Any = None
        self._pool_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(pool_max)  # a burst waits for a free connection instead of failing

    def _get_pool(self) -> Any:
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    from psycopg2.pool import ThreadedConnectionPool

                    self._pool = ThreadedConnectionPool(1, self._pool_max, **_connection_arguments())
        return self._pool

    def fetch(self, dataset: str, doc_ids: Sequence[str]) -> Dict[str, str]:
        """{doc_id: text} for the ids found in the dataset's table. Raises if the database cannot be queried."""
        if not doc_ids:
            return {}
        if self._fetcher is not None:
            return self._fetcher(dataset, doc_ids)
        import psycopg2
        from psycopg2 import sql

        query = sql.SQL("SELECT doc_id, text FROM {} WHERE doc_id = ANY(%s)").format(sql.Identifier(table_name(dataset)))
        pool = self._get_pool()
        with self._slots:
            for attempt in (1, 2):
                conn = pool.getconn()
                broken = False
                try:
                    conn.autocommit = True  # read-only: never leave a transaction open while the connection idles in the pool
                    with conn.cursor() as cursor:
                        cursor.execute(query, (list(doc_ids),))
                        return {row[0]: row[1] for row in cursor.fetchall()}
                except (psycopg2.OperationalError, psycopg2.InterfaceError):
                    broken = True  # e.g. the database restarted while the connection sat in the pool: retry once with a new one
                    if attempt == 2:
                        raise
                finally:
                    pool.putconn(conn, close=broken)
        return {}  # pragma: no cover

    def close(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                try:
                    self._pool.closeall()
                finally:
                    self._pool = None
