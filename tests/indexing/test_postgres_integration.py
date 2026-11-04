"""The SQL layer against a REAL PostgreSQL. Skipped unless TEST_DATABASE_URL is set.

Point TEST_DATABASE_URL at a THROW-AWAY database (for example `CREATE DATABASE ir_test;` on the compose postgres):
the test creates and drops only its own uniquely named table, but it is still a real database.
    TEST_DATABASE_URL=postgresql://ir_user:ir_pass@postgres:5432/ir_test python -m unittest tests.indexing.test_postgres_integration -v
"""
import json
import os
import sys
import unittest
from unittest import mock

from tests.indexing._support import FakeDoc, FakeIrDataset, WorkdirTestCase, fake_preprocessing, have

TEST_URL = os.environ.get("TEST_DATABASE_URL")


@unittest.skipUnless(TEST_URL and have("psycopg2"), "set TEST_DATABASE_URL (and install psycopg2) to run against PostgreSQL")
class PostgresIntegrationTest(WorkdirTestCase):
    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ, {"DATABASE_URL": TEST_URL})  # common.database.get_db_connection() reads it first
        env.start()
        self.addCleanup(env.stop)
        self.dataset = f"zz-test/Idx-{os.getpid()}"
        self.table = f"dataset_zz_test_idx_{os.getpid()}"
        self.addCleanup(self._drop_table)

    def _query(self, sql, params=None):
        from common.database import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
            conn.commit()
            return rows
        finally:
            conn.close()

    def _drop_table(self):
        self._query(f"DROP TABLE IF EXISTS {self.table} CASCADE")

    def test_upload_stream_list_and_delete(self):
        from services.indexing.dataset.db_operations import (
            DatabaseInfoRetriever, DatasetDatabaseDeleter, DatasetUploader,
        )
        from services.indexing.services.build_service import _preprocess_documents

        docs = [FakeDoc("d1", "Heat transfer heat"), FakeDoc("d2", "   "), FakeDoc("d3", "boundary layer flow"),
                FakeDoc("d4", "flow flow"), FakeDoc("d5", "heating")]
        fake_ir = mock.Mock()
        fake_ir.load.return_value = FakeIrDataset(docs)

        # upload: the table name is lower-case, blank documents are skipped, both indexes exist
        with mock.patch("services.indexing.dataset.db_operations.ir_datasets", fake_ir):
            result = DatasetUploader().upload_to_db(self.dataset, batch_size=2)
        self.assertEqual((result["table"], result["inserted"], result["skipped"]), (self.table, 4, 1))
        self.assertEqual(self._query(f"SELECT count(*) FROM {self.table}")[0][0], 4)
        indexes = {row[0] for row in self._query("SELECT indexname FROM pg_indexes WHERE tablename = %s", (self.table,))}
        self.assertIn(f"idx_{self.table}_doc_id", indexes)
        self.assertIn(f"idx_{self.table}_text_gin", indexes)

        # the database overview lists it
        info = DatabaseInfoRetriever().get_info()
        self.assertEqual(info["status"], "healthy")
        listed = [d for d in info["datasets"] if d["table_name"] == self.table]
        self.assertEqual([d["doc_count"] for d in listed], [4])

        # the preprocessing read streams the table through a server-side cursor
        shared = self.work / "shared.jsonl"
        with mock.patch.dict(sys.modules, fake_preprocessing()):
            _preprocess_documents(self.dataset, None, str(shared))
        lines = [json.loads(line) for line in shared.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([d["doc_id"] for d in lines], ["d1", "d3", "d4", "d5"])

        # deleting drops the table
        deleted = DatasetDatabaseDeleter().delete_dataset_from_db(self.dataset)
        self.assertEqual((deleted["status"], deleted["deleted"]), ("success", 4))
        self.assertEqual(self._query("SELECT to_regclass(%s)", (self.table,))[0][0], None)


if __name__ == "__main__":
    unittest.main()
