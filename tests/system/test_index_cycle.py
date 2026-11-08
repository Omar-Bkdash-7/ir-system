"""The full index lifecycle on a dataset that has no indices yet: upload -> build -> serve -> clean up.

Opt-in, because it WRITES (a PostgreSQL table plus an index directory) and needs a downloaded dataset:

    SYSTEM_TEST_INDEX_DATASET=beir/nfcorpus python -m unittest tests.system.test_index_cycle -v

Everything it creates is removed in tearDownModule, even when a test fails. The dataset must already be downloaded
locally (see GET /datasets/local); the test never downloads anything.
"""
import os
import unittest

from tests.system import _stack
from tests.system._stack import call

DATASET = os.environ.get("SYSTEM_TEST_INDEX_DATASET")
STEM_CONFIG = {"use_stemming": True, "remove_stopwords": True}
QUERY = "do cholesterol drugs improve memory"
TOKENS = ["cholesterol", "drug", "improve", "memory"]


def setUpModule():
    _stack.require_stack(("indexing", "retrieval", "api_gateway"))
    if not DATASET:
        raise unittest.SkipTest("set SYSTEM_TEST_INDEX_DATASET (e.g. beir/nfcorpus) to run the index lifecycle")


def tearDownModule():
    if not DATASET:
        return
    call("indexing", "/index/delete-all", method="DELETE", params={"dataset_name": DATASET}, timeout=600)
    call("indexing", "/datasets/db", method="DELETE", params={"dataset_id": DATASET}, timeout=600)


class IndexCycleTest(unittest.TestCase):
    vocabularies = {}

    def build(self, kind, config=None, force=False):
        return call("indexing", f"/index/{kind}/build", {"preprocessing_config": config} if config else None,
                    method="POST", params={"dataset_name": DATASET, "force_rebuild": "true" if force else "false"},
                    timeout=3600)

    def retrieve(self, model, top_k=5):
        return call("retrieval", "/retrieve", {"query_tokens": TOKENS, "query_text": QUERY, "model": model,
                                               "top_k": top_k, "include_text": True, "dataset": DATASET})

    def test_01_the_dataset_can_be_uploaded(self):
        response = call("indexing", "/datasets/upload-to-db", method="POST", params={"dataset_id": DATASET},
                        timeout=3600)
        self.assertEqual(response.status, 200, response.raw[:300])
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertGreater(body["inserted"], 0)

    def test_02_a_default_build_succeeds(self):
        response = self.build("tfidf")
        self.assertEqual(response.status, 200, response.raw[:300])
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertFalse(body["preprocessing_applied"])
        self.assertGreater(body["vocab_size"], 0)
        self.vocabularies["default"] = body["vocab_size"]

    def test_03_a_custom_preprocessing_config_changes_the_index(self):
        response = self.build("tfidf", STEM_CONFIG, force=True)
        self.assertEqual(response.status, 200, response.raw[:300])
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertTrue(body["preprocessing_applied"])
        self.assertLess(body["vocab_size"], self.vocabularies["default"],
                        "the same documents with stemming produced the same vocabulary: preprocessing_config was "
                        "accepted but not applied")
        self.vocabularies["stemmed"] = body["vocab_size"]

    def test_04_the_built_indices_are_served(self):
        self.assertEqual(self.build("bm25", STEM_CONFIG).status, 200)
        stats = call("indexing", "/index/stats", params={"dataset_name": DATASET}).json()
        self.assertEqual({kind: stats["indices"][kind]["status"] for kind in ("tfidf", "bm25")},
                         {"tfidf": "ready", "bm25": "ready"})
        for model in ("tfidf", "bm25"):
            with self.subTest(model=model):
                response = self.retrieve(model)
                self.assertEqual(response.status, 200, response.raw[:300])
                self.assertGreater(response.json()["total"], 0, f"{model} served no document")
        through_gateway = call("api_gateway", "/search", {"query": QUERY, "model": "tfidf", "top_k": 5,
                                                          "dataset": DATASET})
        self.assertEqual(through_gateway.status, 200, through_gateway.raw[:300])
        self.assertGreater(through_gateway.json()["total"], 0)

    def test_05_an_invalid_config_is_refused_and_the_index_survives(self):
        response = self.build("tfidf", {"bogus_key": 1}, force=True)
        self.assertEqual(response.status, 400, response.raw[:300])
        self.assertIn("Invalid preprocessing_config", response.raw)
        self.assertEqual(self.retrieve("tfidf").status, 200, "a refused rebuild damaged the existing index")


if __name__ == "__main__":
    unittest.main()
