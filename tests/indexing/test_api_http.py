"""HTTP level: routes, parameters and status codes of the FastAPI app.

Needs fastapi + httpx (the same packages the service itself needs for its tests); skipped when they are missing."""
import logging
import os
import tempfile
import unittest
from unittest import mock

from tests.indexing._support import FakeDoc, FakeIrDataset, WorkdirTestCase, have

EXPECTED_ROUTES = {
    ("GET", "/health"), ("POST", "/validate"), ("POST", "/datasets/download"), ("GET", "/datasets/local"),
    ("DELETE", "/datasets/local"), ("POST", "/datasets/upload-to-db"),
    ("GET", "/datasets/db-info"), ("GET", "/datasets/metadata"), ("DELETE", "/datasets/db"),
    ("POST", "/datasets/fix-corrupted"), ("GET", "/index/stats"), ("POST", "/index/tfidf/build"),
    ("GET", "/index/tfidf/status"), ("POST", "/index/embedding/build"), ("GET", "/index/embedding/status"),
    ("POST", "/index/bm25/build"), ("GET", "/index/bm25/status"), ("POST", "/index/build-all"),
    ("GET", "/index/inverted/stats"), ("GET", "/index/inverted/term/{term}"), ("GET", "/index/inverted/search"),
    ("GET", "/index/inverted/sample"), ("DELETE", "/index/{index_type}/delete"), ("DELETE", "/index/delete-all"),
}


class _Ir:
    """Stands in for ir_datasets inside dataset/maintenance.py."""

    def __init__(self, known=()):
        self.known = {name: FakeIrDataset([FakeDoc("d1", "text")], queries=["q"], qrels=["r"]) for name in known}

    def load(self, dataset_id):
        if dataset_id not in self.known:
            raise KeyError(dataset_id)
        return self.known[dataset_id]


@unittest.skipUnless(have("fastapi", "httpx"), "FastAPI test client (fastapi + httpx) is not installed")
class ApiTest(WorkdirTestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        previous = os.getcwd()
        # importing main calls setup_logging(), which creates logs/indexing.log relative to the working directory
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
            os.chdir(scratch)
            try:
                from services.indexing.main import app
            finally:
                os.chdir(previous)
        cls.app = app
        cls.client = TestClient(app, raise_server_exceptions=False)
        logging.disable(logging.CRITICAL)  # the service logs every request at INFO
        cls.addClassCleanup(logging.disable, logging.NOTSET)

    def _build_bm25(self, name="demo/ds"):
        from services.indexing.indexers import BM25Indexer
        BM25Indexer(name).build(self.processed()).save()

    # ---- routes -----------------------------------------------------------------------------------------------------
    def test_the_route_table_is_unchanged(self):
        routes = {(m, r.path) for r in self.app.routes for m in (getattr(r, "methods", None) or ())}
        self.assertEqual(EXPECTED_ROUTES - routes, set())

    def test_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    # ---- the five endpoints that used to answer 500 / wrong data -------------------------------------------------------
    def test_the_explorer_endpoints_require_a_dataset_name(self):
        for url in ("/index/stats", "/index/inverted/stats", "/index/inverted/term/flow",
                    "/index/inverted/search?prefix=he", "/index/inverted/sample"):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 422)

    def test_the_explorer_endpoints_work_with_a_dataset_name(self):
        self._build_bm25()
        ds = {"dataset_name": "demo/ds"}
        stats = self.client.get("/index/inverted/stats", params=ds)
        self.assertEqual(stats.status_code, 200)
        self.assertEqual((stats.json()["status"], stats.json()["vocab_size"]), ("success", 8))
        term = self.client.get("/index/inverted/term/flow", params=ds)
        self.assertEqual((term.status_code, term.json()["exists"], term.json()["doc_frequency"]), (200, True, 2))
        search = self.client.get("/index/inverted/search", params={**ds, "prefix": "he"})
        self.assertEqual([r["term"] for r in search.json()["results"]], ["heat", "heating"])
        sample = self.client.get("/index/inverted/sample", params={**ds, "sample_size": 3})
        self.assertEqual(sample.json()["sample_size"], 3)
        overview = self.client.get("/index/stats", params={**ds, "doc_count": 4})
        self.assertEqual(overview.json()["indices"]["bm25"]["status"], "ready")

    def test_an_index_that_does_not_exist_is_not_an_error(self):
        response = self.client.get("/index/inverted/stats", params={"dataset_name": "nothing/here"})
        self.assertEqual((response.status_code, response.json()["status"]), (200, "not_built"))

    def test_an_empty_dataset_name_is_a_client_error(self):
        self.assertEqual(self.client.get("/index/inverted/stats", params={"dataset_name": ""}).status_code, 400)

    # ---- deleting indices ---------------------------------------------------------------------------------------------
    def test_deleting_indices(self):
        self._build_bm25("demo/v1.2")
        sibling = self.work / "data" / "indices" / "keep_me"
        sibling.mkdir(parents=True)
        (sibling / "x.txt").write_text("x")
        ds = {"dataset_name": "demo/v1.2"}

        self.assertEqual(self.client.delete("/index/nope/delete", params=ds).status_code, 400)
        self.assertEqual(self.client.delete("/index/nope/delete", params=ds).json()["detail"], "Invalid index type: nope")
        for bad in ("..", "."):
            self.assertEqual(self.client.delete("/index/delete-all", params={"dataset_name": bad}).status_code, 400)
        self.assertTrue((sibling / "x.txt").exists())

        deleted = self.client.delete("/index/bm25/delete", params=ds)
        self.assertEqual((deleted.status_code, deleted.json()["status"], deleted.json()["count"]), (200, "success", 1))
        everything = self.client.delete("/index/delete-all", params=ds)  # also removes the (now empty) dataset folder
        self.assertEqual((everything.status_code, everything.json()["status"]), (200, "success"))
        gone = self.client.delete("/index/delete-all", params=ds)
        self.assertEqual((gone.status_code, gone.json()["status"]), (200, "not_found"))
        self.assertTrue((sibling / "x.txt").exists())

    # ---- database endpoints -------------------------------------------------------------------------------------------------
    def test_ids_that_cannot_be_table_names_are_a_client_error(self):
        # both handlers wrap every failure into a 500; the unusable id is detected before any database access
        for bad in ("x; drop table y", "a b"):
            with self.subTest(bad=bad):
                self.assertEqual(self.client.post("/datasets/upload-to-db", params={"dataset_id": bad}).status_code, 400)
                self.assertEqual(self.client.delete("/datasets/db", params={"dataset_id": bad}).status_code, 400)

    # ---- build endpoints ---------------------------------------------------------------------------------------------------
    def test_a_failed_build_is_an_http_error_not_a_200(self):
        from services.indexing.services import build_service as builders
        for reason, expected in (("dataset_not_in_database", 400), ("invalid_preprocessing_config", 400),
                                 ("build_failed", 500), (None, 500)):
            with self.subTest(reason=reason):
                failure = {"status": "error", "message": f"boom ({reason})"}
                if reason:
                    failure["reason"] = reason
                with mock.patch.object(builders, "build_bm25_index", return_value=failure), \
                        mock.patch.object(builders, "build_all_indices", return_value=failure):
                    for url in ("/index/bm25/build", "/index/build-all"):
                        response = self.client.post(url, params={"dataset_name": "demo/ds"})
                        self.assertEqual(response.status_code, expected, url)
                        self.assertIn("boom", response.json()["detail"])

    def test_a_skipped_or_finished_build_stays_a_200(self):
        from services.indexing.services import build_service as builders
        for result in ({"status": "skipped", "message": "BM25 index already exists."},
                       {"status": "success", "vocab_size": 8}):
            with self.subTest(status=result["status"]), \
                    mock.patch.object(builders, "build_bm25_index", return_value=result):
                response = self.client.post("/index/bm25/build", params={"dataset_name": "demo/ds"})
                self.assertEqual((response.status_code, response.json()), (200, result))

    # ---- local datasets ---------------------------------------------------------------------------------------------------
    def test_unsafe_dataset_ids_are_rejected_and_nothing_is_deleted(self):
        (self.ir_home / "beir" / "quora").mkdir(parents=True)
        (self.work / "precious.txt").write_text("keep me")
        for bad in ("..", "/", "../x"):
            with self.subTest(bad=bad):
                self.assertEqual(self.client.delete("/datasets/local", params={"dataset_id": bad}).status_code, 400)
                self.assertEqual(self.client.post("/datasets/fix-corrupted", params={"dataset_id": bad}).status_code, 400)
        self.assertTrue((self.work / "precious.txt").exists())
        self.assertTrue((self.ir_home / "beir" / "quora").exists())

    def test_delete_local_dataset(self):
        (self.ir_home / "beir" / "quora").mkdir(parents=True)
        self.assertEqual(self.client.delete("/datasets/local", params={"dataset_id": "beir/quora"}).status_code, 200)
        self.assertEqual(self.client.delete("/datasets/local", params={"dataset_id": "beir/quora"}).status_code, 404)

    def test_fix_corrupted_checks_the_id_before_deleting(self):
        from services.indexing.dataset import maintenance
        (self.ir_home / "beir" / "quora").mkdir(parents=True)
        (self.ir_home / "beir" / "quora" / "broken.tsv").write_text("corrupt")
        with mock.patch.object(maintenance, "ir_datasets", _Ir(known=["beir/quora"])):
            self.assertEqual(self.client.post("/datasets/fix-corrupted", params={"dataset_id": "beir/typo"}).status_code, 500)
            self.assertTrue((self.ir_home / "beir" / "quora" / "broken.tsv").exists())  # unknown id: nothing deleted
            fixed = self.client.post("/datasets/fix-corrupted", params={"dataset_id": "beir/quora"})
        self.assertEqual((fixed.status_code, fixed.json()["status"]), (200, "success"))
        self.assertFalse((self.ir_home / "beir" / "quora" / "broken.tsv").exists())


if __name__ == "__main__":
    unittest.main()
