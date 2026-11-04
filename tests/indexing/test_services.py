"""Status / statistics / inverted-index explorer services and index deletion."""
import random
import unittest
from unittest import mock

import numpy as np

from services.indexing.indexers import BM25Indexer, EmbeddingIndexer, TFIDFIndexer
from services.indexing.services import stats_service
from services.indexing.services.index_admin import delete_all_indices, delete_index
from services.indexing.services.inverted_index_service import (
    get_inverted_index_stats, get_term_info, get_vocabulary_sample, search_terms,
)
from tests.indexing._support import DOCS, WorkdirTestCase


class InvertedIndexServiceTest(WorkdirTestCase):
    def setUp(self):
        super().setUp()
        BM25Indexer("demo/ds").build(self.processed()).save()

    def test_statistics(self):
        stats = get_inverted_index_stats("demo/ds")
        self.assertEqual((stats["status"], stats["vocab_size"], stats["total_docs"]), ("success", 8, 4))
        self.assertEqual((stats["avg_doc_length"], stats["total_postings"]), (3.25, 11))
        self.assertEqual(stats["top_terms_by_df"][0], {"term": "boundary", "doc_freq": 2, "idf": 0.6931})

    def test_term_info(self):
        info = get_term_info("demo/ds", "flow")
        self.assertEqual((info["exists"], info["doc_frequency"], info["total_postings"]), (True, 2, 2))
        self.assertEqual(info["posting_list_sample"], [{"doc_id": "d2", "term_freq": 1}, {"doc_id": "d3", "term_freq": 2}])
        self.assertFalse(get_term_info("demo/ds", "zzz")["exists"])

    def test_prefix_search_and_sample(self):
        found = search_terms("demo/ds", "he", 10)
        self.assertEqual([r["term"] for r in found["results"]], ["heat", "heating"])
        random.seed(0)
        self.assertEqual(get_vocabulary_sample("demo/ds", 3)["sample_size"], 3)

    def test_not_built_and_missing_name(self):
        self.assertEqual(get_inverted_index_stats("nothing/here")["status"], "not_built")
        with self.assertRaisesRegex(ValueError, "dataset_name is required"):
            get_inverted_index_stats("")

    def test_the_index_is_not_unpickled_on_every_call(self):
        import pickle
        real_load, loads = pickle.load, []
        with mock.patch("pickle.load", side_effect=lambda *a, **k: loads.append(1) or real_load(*a, **k)):
            for _ in range(3):
                get_inverted_index_stats("demo/ds")
                get_term_info("demo/ds", "flow")
        self.assertEqual(len(loads), 1)

    def test_a_rebuilt_index_is_noticed(self):
        self.assertEqual(get_inverted_index_stats("demo/ds")["total_docs"], 4)
        BM25Indexer("demo/ds").build(self.processed("b.jsonl", DOCS[:2])).save()
        self.assertEqual(get_inverted_index_stats("demo/ds")["total_docs"], 2)


class StatsServiceTest(WorkdirTestCase):
    def test_overview_of_a_dataset(self):
        BM25Indexer("demo").build(self.processed()).save()
        stats = stats_service.get_index_stats("demo", 4)
        self.assertEqual(stats["dataset"], {"total_docs": 4, "name": "demo"})
        self.assertEqual(stats["indices"]["tfidf"], {"status": "not_built"})
        self.assertEqual(stats["indices"]["embedding"], {"status": "not_built"})
        self.assertEqual(stats["indices"]["bm25"]["status"], "ready")

    def test_tfidf_status(self):
        self.assertEqual(stats_service.get_tfidf_stats("demo"), {"status": "not_built"})
        TFIDFIndexer("demo").build(self.processed()).save()
        stats = stats_service.get_tfidf_stats("demo")
        self.assertEqual((stats["status"], stats["total_docs"]), ("ready", 4))

    def test_embedding_status_does_not_read_the_whole_matrix(self):
        indexer = EmbeddingIndexer("demo")
        indexer.embeddings, indexer.doc_ids = np.ones((4, 3), dtype="float32"), ["d1", "d2", "d3", "d4"]
        indexer.save()
        real_load, modes = np.load, []
        with mock.patch("numpy.load", side_effect=lambda *a, **k: modes.append(k.get("mmap_mode")) or real_load(*a, **k)):
            stats = stats_service.get_embedding_stats("demo")
        self.assertEqual((stats["status"], stats["shape"]), ("ready", [4, 3]))
        self.assertEqual(modes, ["r"])


class IndexAdminTest(WorkdirTestCase):
    def setUp(self):
        super().setUp()
        for kind in (BM25Indexer, TFIDFIndexer):
            kind("demo/v1.2").build(self.processed()).save()  # a dotted id: the old delete endpoint missed it
        self.sibling = self.work / "data" / "indices" / "keep_me"
        self.sibling.mkdir(parents=True)
        (self.sibling / "x.txt").write_text("x")

    def test_delete_one_index_type(self):
        result = delete_index("bm25", "demo/v1.2")
        self.assertEqual((result["status"], result["count"], result["deleted_files"]), ("success", 1, ["bm25_index.pkl"]))
        self.assertFalse(BM25Indexer("demo/v1.2").exists())
        self.assertTrue(TFIDFIndexer("demo/v1.2").exists())
        self.assertEqual(delete_index("bm25", "demo/v1.2")["status"], "not_found")

    def test_delete_everything_of_one_dataset(self):
        result = delete_all_indices("demo/v1.2")
        self.assertEqual((result["status"], result["deleted_count"]), ("success", 4))
        self.assertFalse(TFIDFIndexer("demo/v1.2").exists())
        self.assertEqual(delete_all_indices("demo/v1.2")["status"], "not_found")

    def test_deleting_clears_the_cache(self):
        self.assertEqual(get_inverted_index_stats("demo/v1.2")["status"], "success")
        delete_index("bm25", "demo/v1.2")
        self.assertEqual(get_inverted_index_stats("demo/v1.2")["status"], "not_built")

    def test_unsafe_names_are_refused_and_nothing_is_touched(self):
        for bad in ("", ".", ".."):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                delete_all_indices(bad)
        with self.assertRaisesRegex(ValueError, "Invalid index type: nope"):
            delete_index("nope", "demo/v1.2")
        self.assertTrue((self.sibling / "x.txt").exists())
        self.assertTrue(BM25Indexer("demo/v1.2").exists())


if __name__ == "__main__":
    unittest.main()
