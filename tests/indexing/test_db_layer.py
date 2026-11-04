"""The PostgreSQL layer, run against a recording stand-in for the connection (no database needed).

What is checked: the SQL that is sent (table names, schema), that connections are always closed, that unusable ids
never reach the database, that the preprocessing read is streamed with a server-side cursor, and the build rules."""
import json
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from services.indexing.dataset.db_operations import DatabaseInfoRetriever, DatasetDatabaseDeleter, DatasetUploader
from services.indexing.dataset.metadata_manager import DatasetMetadataManager
from services.indexing.indexers import BM25Indexer, EmbeddingIndexer, TFIDFIndexer
from services.indexing.services import build_service
from services.indexing.services.build_service import (
    BM25IndexBuilder, _check_dataset_in_database, _preprocess_documents, build_all_indices,
)
from services.indexing.storage import try_lock
from tests.indexing._support import (
    FakeDatabase, FakeDoc, FakeIrDataset, WorkdirTestCase, fake_preprocessing, write_processed,
)


def _execute_batch(cur, sql, argslist, page_size=100):
    """psycopg2.extras.execute_batch needs a real cursor (mogrify); the tests only need to see the statements."""
    for args in argslist:
        cur.execute(sql, args)


class UploaderTest(WorkdirTestCase):
    def _upload(self, dataset_id, docs, db, **kwargs):
        fake_ir = mock.Mock()
        fake_ir.load.return_value = FakeIrDataset(docs)
        with mock.patch("services.indexing.dataset.db_operations.get_db_connection", db.connect), \
                mock.patch("services.indexing.dataset.db_operations.ir_datasets", fake_ir), \
                mock.patch("services.indexing.dataset.db_operations.execute_batch", _execute_batch):
            return DatasetUploader().upload_to_db(dataset_id, **kwargs)

    def test_creates_a_lower_case_table_with_the_original_schema(self):
        db = FakeDatabase()
        docs = [FakeDoc("d1", "alpha"), FakeDoc("d2", "   "), FakeDoc("d3", "gamma")]
        result = self._upload("Demo/V1.2", docs, db)
        self.assertEqual((result["status"], result["inserted"], result["skipped"]), ("success", 2, 1))
        self.assertEqual(result["table"], "dataset_demo_v1_2")
        sql = db.statements
        self.assertTrue(any(s.startswith("CREATE TABLE dataset_demo_v1_2 ( doc_id VARCHAR(255) PRIMARY KEY") for s in sql))
        self.assertIn("CREATE INDEX idx_dataset_demo_v1_2_doc_id ON dataset_demo_v1_2(doc_id)", sql)
        self.assertTrue(any("idx_dataset_demo_v1_2_text_gin" in s and "USING gin(to_tsvector('english', text))" in s for s in sql))
        self.assertEqual(len([s for s in sql if s.startswith("INSERT INTO dataset_demo_v1_2")]), 2)
        self.assertTrue(all(c.closed for c in db.connections))

    def test_an_existing_table_is_replaced(self):
        db = FakeDatabase(tables=["dataset_demo"])
        self._upload("demo", [FakeDoc("d1", "alpha")], db)
        sql = db.statements
        self.assertLess(sql.index("DROP TABLE IF EXISTS dataset_demo CASCADE"),
                        next(i for i, s in enumerate(sql) if s.startswith("CREATE TABLE")))

    def test_documents_are_inserted_in_batches(self):
        db = FakeDatabase()
        inserted = []
        real = _execute_batch
        with mock.patch("services.indexing.dataset.db_operations.execute_batch",
                        side_effect=lambda cur, sql, args, page_size=100: inserted.append(list(args)) or real(cur, sql, [], page_size)):
            fake_ir = mock.Mock()
            fake_ir.load.return_value = FakeIrDataset([FakeDoc(f"d{i}", "t") for i in range(5)])
            with mock.patch("services.indexing.dataset.db_operations.get_db_connection", db.connect), \
                    mock.patch("services.indexing.dataset.db_operations.ir_datasets", fake_ir):
                DatasetUploader().upload_to_db("demo", batch_size=2)
        self.assertEqual([len(b) for b in inserted], [2, 2, 1])

    def test_ids_that_are_not_valid_table_names_never_reach_the_database(self):
        db = FakeDatabase()
        for bad in ("x; drop table y", "a b", "it's"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._upload(bad, [FakeDoc("d1", "alpha")], db)
        self.assertEqual(db.connections, [])  # not even a connection was opened

    def test_a_failed_upload_is_cleaned_up_and_the_connection_is_closed(self):
        db = FakeDatabase()

        def docs():
            yield FakeDoc("d1", "alpha")
            raise RuntimeError("source file is broken")

        fake_ir = mock.Mock()
        fake_ir.load.return_value = mock.Mock(docs_iter=lambda: docs())
        with mock.patch("services.indexing.dataset.db_operations.get_db_connection", db.connect), \
                mock.patch("services.indexing.dataset.db_operations.ir_datasets", fake_ir), \
                mock.patch("services.indexing.dataset.db_operations.execute_batch", _execute_batch), \
                self.assertRaises(RuntimeError):
            DatasetUploader().upload_to_db("demo")
        conn = db.connections[0]
        self.assertEqual(conn.rollbacks, 1)
        self.assertEqual(db.statements[-1], "DROP TABLE IF EXISTS dataset_demo CASCADE")  # the half-created table is removed
        self.assertTrue(conn.closed)


class DatabaseInfoTest(WorkdirTestCase):
    def test_dataset_ids_are_exact_when_the_metadata_knows_them(self):
        DatasetMetadataManager().save_metadata({}, "nano-beir/quora")
        db = FakeDatabase(tables=["dataset_cranfield", "dataset_nano_beir_quora"],
                          rows={"dataset_cranfield": [("a", "b")] * 3, "dataset_nano_beir_quora": [("a", "b")] * 5})
        with mock.patch("services.indexing.dataset.db_operations.get_db_connection", db.connect):
            info = DatabaseInfoRetriever().get_info()
        self.assertEqual(info["status"], "healthy")
        self.assertEqual([(d["dataset_id"], d["doc_count"]) for d in info["datasets"]],
                         [("cranfield", 3), ("nano-beir/quora", 5)])  # was "nano/beir/quora"
        self.assertEqual((info["total_tables"], info["total_documents"]), (2, 8))
        self.assertTrue(db.connections[0].closed)


class DatabaseDeleterTest(WorkdirTestCase):
    def _delete(self, dataset_id, db):
        with mock.patch("services.indexing.dataset.db_operations.get_db_connection", db.connect):
            return DatasetDatabaseDeleter().delete_dataset_from_db(dataset_id)

    def test_drops_the_lower_case_table(self):
        db = FakeDatabase(tables=["dataset_nano_beir_quora"], rows={"dataset_nano_beir_quora": [("a", "b")] * 4})
        result = self._delete("Nano-Beir/Quora", db)
        self.assertEqual((result["status"], result["deleted"], result["table"]), ("success", 4, "dataset_nano_beir_quora"))
        self.assertIn("DROP TABLE dataset_nano_beir_quora CASCADE", db.statements)
        self.assertTrue(db.connections[0].closed)

    def test_unknown_dataset_and_unusable_ids(self):
        db = FakeDatabase()
        self.assertEqual(self._delete("nothing", db)["status"], "not_found")
        self.assertTrue(db.connections[0].closed)
        with self.assertRaises(ValueError):
            self._delete("x; drop table y", db)
        self.assertEqual(len(db.connections), 1)  # the unusable id never opened a connection


class CheckDatasetInDatabaseTest(WorkdirTestCase):
    def _check(self, name, table, db):
        with mock.patch("common.database.get_db_connection", db.connect):
            return _check_dataset_in_database(name, table)

    def test_uploaded_dataset_passes_and_the_connection_is_closed(self):
        db = FakeDatabase(tables=["dataset_demo"])
        self._check("demo", "dataset_demo", db)
        self.assertTrue(db.connections[0].closed)  # `with conn:` alone never closed it

    def test_missing_dataset_lists_the_available_ones_with_their_exact_ids(self):
        DatasetMetadataManager().save_metadata({}, "nano-beir/quora")
        db = FakeDatabase(tables=["dataset_nano_beir_quora"])
        with self.assertRaises(FileNotFoundError) as caught:
            self._check("demo", "dataset_demo", db)
        message = str(caught.exception)
        self.assertIn("Dataset 'demo' is not uploaded to the database.", message)
        self.assertIn("Expected database table: dataset_demo", message)
        self.assertIn("Available datasets in database: nano-beir/quora", message)
        self.assertTrue(db.connections[0].closed)


class PreprocessDocumentsTest(WorkdirTestCase):
    ROWS = [("d1", "Heat transfer heat"), ("d2", "   "), ("d3", "Flow layer"), ("d4", "123 456")]

    def _run(self, db, shared, **kwargs):
        with mock.patch("common.database.get_db_connection", db.connect), \
                mock.patch.dict(sys.modules, fake_preprocessing(**kwargs)):
            return _preprocess_documents("demo", None, str(shared))

    def test_documents_are_streamed_with_a_server_side_cursor(self):
        db = FakeDatabase(tables=["dataset_demo"], rows={"dataset_demo": self.ROWS})
        shared = self.work / "shared.jsonl"
        self.assertEqual(self._run(db, shared), str(shared))
        lines = [json.loads(line) for line in shared.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([d["doc_id"] for d in lines], ["d1", "d3"])  # empty text and token-less documents are skipped
        self.assertEqual(lines[0], {"doc_id": "d1", "text": "Heat transfer heat", "tokens": ["heat", "transfer", "heat"]})
        # two connections: one for the "is the dataset uploaded?" check, one for the read itself
        named = [c for conn in db.connections for c in conn.cursors if c.name]  # a named cursor is a server-side one
        self.assertEqual([c.name for c in named], ["indexing_preprocess"])
        self.assertEqual(named[0].itersize, build_service.SERVER_CURSOR_ROWS)
        self.assertTrue(all(conn.closed for conn in db.connections))

    def test_a_failure_removes_the_partial_file_and_closes_the_connection(self):
        db = FakeDatabase(tables=["dataset_demo"], rows={"dataset_demo": self.ROWS})
        shared = self.work / "shared.jsonl"
        with self.assertRaisesRegex(RuntimeError, "preprocessing exploded"):
            self._run(db, shared, fail_on="Flow")
        self.assertFalse(shared.exists())
        self.assertTrue(db.connections[-1].closed)

    def test_a_dataset_that_was_not_uploaded_is_reported_before_any_file_is_created(self):
        db = FakeDatabase(tables=[])
        shared = self.work / "shared.jsonl"
        with self.assertRaises(FileNotFoundError):
            self._run(db, shared)
        self.assertFalse(shared.exists())


class BuildRulesTest(WorkdirTestCase):
    def _all_indices_exist(self, dataset="demo"):
        processed = self.processed()
        BM25Indexer(dataset).build(processed).save()
        TFIDFIndexer(dataset).build(processed).save()
        emb = EmbeddingIndexer(dataset)
        emb.embeddings, emb.doc_ids = np.ones((4, 3), dtype="float32"), ["d1", "d2", "d3", "d4"]
        emb.save()

    def test_build_all_skips_the_expensive_preprocessing_when_nothing_has_to_be_built(self):
        self._all_indices_exist()
        with mock.patch.object(build_service, "_preprocess_documents") as preprocess, \
                mock.patch.object(build_service, "_check_dataset_in_database") as check:
            result = build_all_indices("demo")
        preprocess.assert_not_called()
        check.assert_called_once()  # "is the dataset uploaded?" is still verified
        self.assertEqual(result["status"], "success")
        self.assertTrue(all(r["status"] == "skipped" for r in result["indices"].values()), result)

    def test_force_rebuild_does_preprocess(self):
        self._all_indices_exist()

        def fake_preprocess(dataset_name, preprocessing_config, shared_path):
            return write_processed(Path(shared_path))

        with mock.patch.object(build_service, "_preprocess_documents", side_effect=fake_preprocess) as preprocess, \
                mock.patch.object(build_service, "_check_dataset_in_database"), \
                mock.patch.object(EmbeddingIndexer, "build", side_effect=RuntimeError("no model in tests")):
            result = build_all_indices("demo", force_rebuild=True)
        preprocess.assert_called_once()
        self.assertEqual(result["indices"]["tfidf"]["status"], "success")
        self.assertEqual(result["indices"]["bm25"]["status"], "success")
        self.assertEqual(result["indices"]["embedding"]["status"], "error")  # the stand-in "model" fails on purpose
        self.assertFalse((self.work / "data" / "processed" / "demo_shared.jsonl").exists())  # shared file is cleaned up

    def test_a_second_identical_build_is_refused_while_one_is_running(self):
        for label, call in (
            ("BM25", lambda: BM25IndexBuilder().build("demo", preprocessed_path=self.processed())),
            ("build-all", lambda: build_all_indices("demo")),
        ):
            key = "all:demo" if label == "build-all" else f"{label}:demo"
            with self.subTest(label=label), try_lock(key) as held:
                self.assertTrue(held)
                result = call()
            self.assertEqual(result["status"], "skipped")
            self.assertIn("already running", result["message"])
        self.assertFalse(BM25Indexer("demo").exists())  # nothing was built by the refused requests

    def test_builders_refuse_names_that_are_missing_or_unsafe(self):
        for bad in ("", ".."):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                BM25IndexBuilder().build(bad, preprocessed_path=self.processed())

    def test_the_reported_index_size_is_the_real_size(self):
        # the old code looked at the YAML path TEMPLATE ("data/indices/{dataset}/bm25/...") which never exists: always 0
        rng = random.Random(1)
        vocab = [f"w{i}" for i in range(2000)]
        docs = [{"doc_id": f"d{i}", "tokens": [rng.choice(vocab) for _ in range(60)]} for i in range(3000)]
        processed = write_processed(self.work / "data" / "processed" / "big.jsonl", docs)
        result = BM25IndexBuilder().build("demo", force_rebuild=True, preprocessed_path=processed)
        self.assertEqual(result["status"], "success", result)
        real_size = Path(BM25Indexer("demo").index_path).stat().st_size / (1024 * 1024)
        self.assertGreater(result["index_size_mb"], 0.1)
        self.assertAlmostEqual(result["index_size_mb"], round(real_size, 2), places=2)


if __name__ == "__main__":
    unittest.main()
