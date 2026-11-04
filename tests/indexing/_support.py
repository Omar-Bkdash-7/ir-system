"""Shared helpers for the indexing tests. Standard library only: run them with `python -m unittest` (no pytest needed)
or with pytest - both work. Every test runs in a fresh temporary directory, so no real data is ever touched."""
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

# A tiny preprocessed corpus (the JSONL the indexers read): doc_id + tokens (+ text for the embedding indexer)
DOCS = [
    {"doc_id": "d1", "text": "heat transfer heat boundary", "tokens": ["heat", "transfer", "heat", "boundary"]},
    {"doc_id": "d2", "text": "boundary layer flow", "tokens": ["boundary", "layer", "flow"]},
    {"doc_id": "d3", "text": "heating flow flow layer x", "tokens": ["heating", "flow", "flow", "layer", "x"]},
    {"doc_id": "d4", "text": "a", "tokens": ["a"]},
]


def have(*modules: str) -> bool:
    """True when every module can be imported (used to skip the tests that need FastAPI / psycopg2 / ...)."""
    return all(importlib.util.find_spec(m) is not None for m in modules)


def write_processed(path: Path, docs=None) -> str:
    docs = DOCS if docs is None else docs
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    return str(path)


class WorkdirTestCase(unittest.TestCase):
    """Runs every test inside its own temporary working directory.

    The service uses relative paths (data/indices, data/metadata, data/processed) and IR_DATASETS_HOME, so after the
    chdir / env change below everything lands in the temporary directory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="indexing_test_")
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name).resolve()
        previous = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, previous)
        self.ir_home = self.work / "ir_home"
        self.ir_home.mkdir()
        env = mock.patch.dict(os.environ, {"IR_DATASETS_HOME": str(self.ir_home)})
        env.start()
        self.addCleanup(env.stop)
        self._clear_cache()
        self.addCleanup(self._clear_cache)

    @staticmethod
    def _clear_cache():
        try:
            from services.indexing.services.index_cache import clear_index_cache
        except ImportError:  # running the tests against the old code
            return
        clear_index_cache()

    def processed(self, name="processed.jsonl", docs=None) -> str:
        return write_processed(self.work / "data" / "processed" / name, docs)


# ---------------------------------------------------------------------------------------------------------------------
# A recording stand-in for a PostgreSQL connection: lets the SQL layer run (and be checked) without a database.
class FakeCursor:
    def __init__(self, conn, name=None):
        self.conn, self.name, self.itersize, self.closed, self._rows = conn, name, None, False, []

    def execute(self, sql, params=None):
        sql = " ".join(str(sql).split())
        self.conn.statements.append(sql)
        self.conn.cursor_names.append(self.name)
        self._rows = list(self.conn.respond(sql, params))

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def __iter__(self):
        while self._rows:
            yield self._rows.pop(0)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeConnection:
    def __init__(self, respond):
        self.respond = respond
        self.statements, self.cursor_names, self.cursors = [], [], []
        self.commits = self.rollbacks = 0
        self.closed = False

    def cursor(self, name=None):
        cur = FakeCursor(self, name)
        self.cursors.append(cur)
        return cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True

    # like psycopg2: `with conn:` ends the transaction but does NOT close the connection
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDatabase:
    """tables: names of the tables that exist; rows: {table: [(doc_id, text), ...]}."""

    def __init__(self, tables=(), rows=None):
        self.tables, self.rows, self.connections = list(tables), dict(rows or {}), []

    def connect(self):
        conn = FakeConnection(self._respond)
        self.connections.append(conn)
        return conn

    @property
    def statements(self):
        return [s for c in self.connections for s in c.statements]

    def _respond(self, sql, params):
        if sql.startswith("SELECT EXISTS"):
            return [(params[0] in self.tables,)]
        if sql.startswith("SELECT table_name FROM information_schema.tables"):
            like = params[0] if params else "dataset_%"
            prefix = like.rstrip("%")
            return [(t,) for t in sorted(self.tables) if t.startswith(prefix)]
        m = re.match(r"SELECT COUNT\(\*\) FROM (\w+)", sql)
        if m:
            return [(len(self.rows.get(m.group(1), [])),)]
        if sql.startswith("SELECT pg_size_pretty"):
            return [("16 kB",)]
        m = re.match(r"SELECT doc_id, text FROM (\w+) ORDER BY doc_id", sql)
        if m:
            return list(self.rows.get(m.group(1), []))
        return []


class FakeDoc:
    def __init__(self, doc_id, text):
        self.doc_id, self.text = doc_id, text


class FakeIrDataset:
    def __init__(self, docs=(), queries=(), qrels=()):
        self._docs, self._queries, self._qrels = list(docs), list(queries), list(qrels)

    def docs_iter(self):
        return iter(self._docs)

    def queries_iter(self):
        return iter(self._queries)

    def qrels_iter(self):
        return iter(self._qrels)


def fake_preprocessing(fail_on=None):
    """sys.modules entries that replace services.preprocessing for the duration of a test."""
    module = types.ModuleType("services.preprocessing.preprocessing_service")

    class PreprocessingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def preprocess(text, config):
        if fail_on and fail_on in text:
            raise RuntimeError("preprocessing exploded")
        return [w.lower() for w in text.split() if w.isalpha()]

    module.preprocess, module.PreprocessingConfig = preprocess, PreprocessingConfig
    return {"services.preprocessing": types.ModuleType("services.preprocessing"),
            "services.preprocessing.preprocessing_service": module}
