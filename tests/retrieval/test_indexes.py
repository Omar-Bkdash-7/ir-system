import json
import pickle
import unittest
from unittest import mock

import numpy as np

from common.exceptions import IndexNotFoundError, RetrievalError
from services.retrieval.core import compact, indexes
from services.retrieval.core.compact import CompactCache
from services.retrieval.core.indexes import SearchOptions, load_index
from services.retrieval.core.query import QueryContext
from tests.retrieval.helpers import (TOY_DOCS, FakeEncoder, bm25_reference, hash_embedding, project_dir, tokens_of,
                                     write_bm25, write_embeddings, write_tfidf)

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False

OPTS = SearchOptions(1.5, 0.75)


def query(tokens=(), text="", encoder=None, mode="preprocessed"):
    return QueryContext(text=text, tokens=list(tokens), encoder=encoder, embedding_query_mode=mode, preprocess=tokens_of)


class CompactCacheTests(unittest.TestCase):
    def test_publish_reload_and_prune_old_versions(self):
        with project_dir() as root:
            pkl = write_bm25(root, "toy", TOY_DOCS)
            cache = CompactCache(root / "cache")
            stamp = compact.source_stamp([pkl])
            key = compact.stamp_key(stamp)
            self.assertIsNone(cache.load("toy", "bm25", key))
            payload = compact.build_bm25_payload(pkl)
            self.assertIsNotNone(cache.save("toy", "bm25", key, payload, stamp))
            loaded = cache.load("toy", "bm25", key)
            self.assertEqual(loaded.json["terms"], payload.json["terms"])
            np.testing.assert_array_equal(loaded.arrays["doc_idx"], payload.arrays["doc_idx"])
            # a different source version gets a new key; publishing it removes the old directory
            write_bm25(root, "toy", {**TOY_DOCS, "d9": "a brand new document"})
            stamp2 = compact.source_stamp([pkl])
            key2 = compact.stamp_key(stamp2)
            self.assertNotEqual(key, key2)
            cache.save("toy", "bm25", key2, compact.build_bm25_payload(pkl), stamp2)
            self.assertEqual([p.name for p in (root / "cache" / "toy").iterdir()], [f"bm25-{key2}"])

    def test_directory_without_manifest_is_ignored(self):
        with project_dir() as root:
            cache = CompactCache(root / "cache")
            (root / "cache" / "toy" / "bm25-abc").mkdir(parents=True)
            self.assertIsNone(cache.load("toy", "bm25", "abc"))

    def test_unwritable_cache_keeps_arrays_in_memory(self):
        with project_dir() as root:
            write_bm25(root, "toy", TOY_DOCS)
            blocker = root / "not_a_directory"
            blocker.write_text("x")
            index = load_index("bm25", "toy", embedding_mmap=True, cache=CompactCache(blocker / "cache"))
            self.assertEqual(index.n_docs, len(TOY_DOCS))

    def test_disabled_cache(self):
        with project_dir() as root:
            write_bm25(root, "toy", TOY_DOCS)
            index = load_index("bm25", "toy", embedding_mmap=True, cache=CompactCache(None))
            self.assertEqual(index.n_docs, len(TOY_DOCS))

    def test_second_load_does_not_rebuild(self):
        with project_dir() as root:
            write_bm25(root, "toy", TOY_DOCS)
            cache = CompactCache(root / "cache")
            load_index("bm25", "toy", embedding_mmap=True, cache=cache)
            with mock.patch.object(indexes, "build_bm25_payload", side_effect=AssertionError("rebuilt")):
                self.assertEqual(load_index("bm25", "toy", embedding_mmap=True, cache=cache).n_docs, len(TOY_DOCS))


class Bm25IndexTests(unittest.TestCase):
    def setUp(self):
        self._ctx = project_dir()
        self.root = self._ctx.__enter__()
        write_bm25(self.root, "toy", TOY_DOCS)
        self.index = load_index("bm25", "toy", embedding_mmap=True, cache=CompactCache(self.root / "cache"))

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def search(self, tokens, k=10, opts=OPTS):
        hits = self.index.search(self.index.prepare(query(tokens)), k, opts)
        return dict(zip(self.index.ids_for(hits.rows), hits.scores.tolist())), self.index.ids_for(hits.rows)

    def test_scores_match_an_independent_implementation(self):
        for tokens in (["boundary", "layer"], ["flat", "plate", "flow"], ["wing"], ["heat", "heat", "layer"], ["viscous", "unknownword"]):
            for k1, b in ((1.5, 0.75), (0.9, 0.2), (2.5, 1.0), (1.5, 0.0)):
                got, order = self.search(tokens, 100, SearchOptions(k1, b))
                expected = bm25_reference(TOY_DOCS, tokens, k1, b)
                self.assertEqual(set(got), set(expected), (tokens, k1, b))
                for doc_id, score in expected.items():
                    self.assertAlmostEqual(got[doc_id], score, places=12)
                self.assertEqual(order, sorted(order, key=lambda d: (-got[d], list(TOY_DOCS).index(d))))

    def test_repeated_query_term_counts_again(self):
        once, _ = self.search(["heat"])
        twice, _ = self.search(["heat", "heat"])
        for doc_id in once:
            self.assertAlmostEqual(twice[doc_id], 2 * once[doc_id], places=12)

    def test_unknown_terms_match_nothing(self):
        self.assertEqual(self.search(["zzz", "qqq"])[0], {})

    def test_top_k_truncates(self):
        self.assertEqual(len(self.search(["boundary", "layer", "flow", "heat"], 2)[1]), 2)

    def test_score_rows_and_unknown_ids(self):
        plan = self.index.prepare(query(["boundary", "layer"]))
        rows = self.index.rows_for(["d4", "not-in-index", "d3"])
        self.assertEqual(rows.tolist()[1], -1)
        scores = self.index.score(plan, rows, OPTS)
        reference = bm25_reference(TOY_DOCS, ["boundary", "layer"])
        self.assertAlmostEqual(scores[0], reference["d4"], places=12)
        self.assertEqual(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)  # d3 contains neither word

    def test_stats(self):
        stats = self.index.stats()
        self.assertEqual((stats["total_docs"], stats["postings"] > 0), (len(TOY_DOCS), True))


class Bm25LegacyDataTests(unittest.TestCase):
    def test_idf_is_computed_from_document_frequencies_when_missing(self):
        with project_dir() as root:
            write_bm25(root, "toy", TOY_DOCS, include_idf=False)
            index = load_index("bm25", "toy", embedding_mmap=True, cache=CompactCache(None))
            hits = index.search(index.prepare(query(["boundary", "layer"])), 10, OPTS)
            expected = bm25_reference(TOY_DOCS, ["boundary", "layer"])
            for doc_id, score in zip(index.ids_for(hits.rows), hits.scores.tolist()):
                self.assertAlmostEqual(score, expected[doc_id], places=12)

    def test_postings_for_ids_missing_from_doc_ids_are_still_scored(self):
        with project_dir() as root:
            pkl = write_bm25(root, "toy", TOY_DOCS)
            data = pickle.loads(pkl.read_bytes())
            data["inverted_index"]["ghost"] = [("zz-extra", 2)]
            data["doc_frequencies"]["ghost"] = 1
            data["idf_scores"]["ghost"] = 1.0
            pkl.write_bytes(pickle.dumps(data))
            index = load_index("bm25", "toy", embedding_mmap=True, cache=CompactCache(None))
            hits = index.search(index.prepare(query(["ghost"])), 5, OPTS)
            self.assertEqual(index.ids_for(hits.rows), ["zz-extra"])
            self.assertGreater(hits.scores[0], 0)

    def test_duplicate_doc_ids_are_merged(self):
        with project_dir() as root:
            pkl = write_bm25(root, "toy", TOY_DOCS)
            data = pickle.loads(pkl.read_bytes())
            data["doc_ids"] = data["doc_ids"] + ["d1"]
            pkl.write_bytes(pickle.dumps(data))
            index = load_index("bm25", "toy", embedding_mmap=True, cache=CompactCache(None))
            self.assertEqual(index.n_docs, len(TOY_DOCS))


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn is not installed")
class TfidfIndexTests(unittest.TestCase):
    def build(self, **vectorizer_args):
        self._ctx = project_dir()
        self.root = self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__, None, None, None)
        self.vectorizer, self.matrix = write_tfidf(self.root, "toy", TOY_DOCS, **vectorizer_args)
        self.index = load_index("tfidf", "toy", embedding_mmap=True, cache=CompactCache(self.root / "cache"))

    def reference(self, tokens):
        from sklearn.metrics.pairwise import cosine_similarity
        q = self.vectorizer.transform([" ".join(tokens)])
        return cosine_similarity(q, self.matrix).ravel() if q.nnz else np.zeros(self.matrix.shape[0])

    def test_native_query_vector_equals_sklearn_transform(self):
        self.build()
        self.assertTrue(self.index._meta["vectorizer"]["native"])
        words = sorted({w for text in TOY_DOCS.values() for w in tokens_of(text)}) + ["unknown", "x", "A", "Flat-Plate"]
        rng = np.random.default_rng(3)
        for _ in range(60):
            tokens = [words[i] for i in rng.integers(0, len(words), int(rng.integers(1, 7)))]
            columns, weights = self.index._vectorize(tokens)
            expected = self.vectorizer.transform([" ".join(tokens)]).tocsr()
            expected.sort_indices()  # sklearn does not guarantee column order; compare canonical CSR rows
            if expected.nnz == 0:
                self.assertEqual(columns.size, 0)
                continue
            order = np.argsort(columns)
            np.testing.assert_array_equal(columns[order], expected.indices)
            np.testing.assert_allclose(weights[order], expected.data, rtol=1e-12)

    def test_search_equals_cosine_similarity_ranking(self):
        self.build()
        for tokens in (["boundary", "layer"], ["flat", "plate"], ["wing", "flutter", "wind"], ["unknown"], ["heat"]):
            expected = self.reference(tokens)
            hits = self.index.search(self.index.prepare(query(tokens)), 100, OPTS)
            ids = list(TOY_DOCS)
            got = dict(zip(self.index.ids_for(hits.rows), hits.scores.tolist()))
            self.assertEqual(set(got), {ids[i] for i in np.flatnonzero(expected > 0)})
            for doc_id, score in got.items():
                self.assertAlmostEqual(score, expected[ids.index(doc_id)], places=12)

    def test_score_rows(self):
        self.build()
        tokens = ["boundary", "layer"]
        rows = self.index.rows_for(["d4", "nope", "d2"])
        scores = self.index.score(self.index.prepare(query(tokens)), rows, OPTS)
        expected = self.reference(tokens)
        self.assertAlmostEqual(scores[0], expected[3], places=12)
        self.assertEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], expected[1], places=12)

    def test_unusual_vectorizer_settings_fall_back_to_sklearn(self):
        self.build(ngram_range=(1, 2))
        self.assertFalse(self.index._meta["vectorizer"]["native"])
        tokens = ["boundary", "layer", "flow"]
        expected = self.reference(tokens)
        hits = self.index.search(self.index.prepare(query(tokens)), 100, OPTS)
        for doc_id, score in zip(self.index.ids_for(hits.rows), hits.scores.tolist()):
            self.assertAlmostEqual(score, expected[list(TOY_DOCS).index(doc_id)], places=12)

    def test_unnormalised_matrix_is_normalised_when_the_cache_is_built(self):
        self.build()
        import scipy.sparse as sp
        broken = self.matrix * 3.0
        sp.save_npz(self.root / "data" / "indices" / "toy" / "tfidf" / "tfidf_matrix.npz", broken)
        index = load_index("tfidf", "toy", embedding_mmap=True, cache=CompactCache(None))
        hits = index.search(index.prepare(query(["boundary", "layer"])), 5, OPTS)
        self.assertLessEqual(float(hits.scores.max()), 1.0 + 1e-9)

    def test_matrix_and_doc_ids_from_different_builds_are_rejected(self):
        self.build()
        path = self.root / "data" / "indices" / "toy" / "tfidf" / "tfidf_doc_ids.json"
        path.write_text(json.dumps(["only", "two"]))
        with self.assertRaises(RetrievalError):
            load_index("tfidf", "toy", embedding_mmap=True, cache=CompactCache(None))


class DenseIndexTests(unittest.TestCase):
    def setUp(self):
        self._ctx = project_dir()
        self.root = self._ctx.__enter__()
        self.matrix = write_embeddings(self.root, "toy", TOY_DOCS)
        self.index = load_index("embedding", "toy", embedding_mmap=True, cache=CompactCache(None))
        self.encoder = FakeEncoder()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def test_search_equals_brute_force(self):
        q = query(text="heat transfer boundary layer", encoder=self.encoder)
        hits = self.index.search(self.index.prepare(q), 4, OPTS)
        vector = hash_embedding(self.encoder.calls[0])
        expected = self.matrix @ vector
        order = sorted(range(len(expected)), key=lambda i: (-expected[i], i))[:4]
        self.assertEqual(hits.rows.tolist(), order)
        np.testing.assert_allclose(hits.scores, expected[order], rtol=1e-6)

    def test_score_rows_with_unknown_documents(self):
        q = query(text="wing flutter", encoder=self.encoder)
        plan = self.index.prepare(q)
        scores = self.index.score(plan, self.index.rows_for(["d3", "missing"]), OPTS)
        self.assertAlmostEqual(scores[0], float(self.matrix[2] @ plan), places=6)
        self.assertEqual(scores[1], 0.0)

    def test_dimension_mismatch_is_reported(self):
        with self.assertRaises(RetrievalError):
            self.index.prepare(query(text="x y z", encoder=FakeEncoder(dim=8)))

    def test_non_finite_scores_are_rejected(self):
        bad = np.full(self.index.dim, np.nan, dtype=np.float32)
        with self.assertRaises(RetrievalError):
            self.index.search(bad, 3, OPTS)

    def test_float64_matrix_is_converted(self):
        path = self.root / "data" / "indices" / "toy" / "embedding" / "embeddings.npy"
        np.save(path, self.matrix.astype(np.float64))
        index = load_index("embedding", "toy", embedding_mmap=True, cache=CompactCache(None))
        self.assertEqual(index._matrix.dtype, np.float32)

    def test_embeddings_and_ids_must_agree(self):
        (self.root / "data" / "indices" / "toy" / "embedding" / "doc_id_map.json").write_text(json.dumps(["a"]))
        with self.assertRaises(RetrievalError):
            load_index("embedding", "toy", embedding_mmap=True, cache=CompactCache(None))


class MissingIndexTests(unittest.TestCase):
    def test_missing_files_are_reported_with_their_path(self):
        with project_dir():
            for kind in ("bm25", "tfidf", "embedding"):
                with self.assertRaises(IndexNotFoundError) as caught:
                    load_index(kind, "nope", embedding_mmap=True, cache=CompactCache(None))
                self.assertIn("nope", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
