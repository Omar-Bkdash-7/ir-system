import unittest

import numpy as np

from common.exceptions import IndexNotFoundError, InvalidQueryError, RetrievalError
from services.retrieval.config import RetrievalSettings
from services.retrieval.core.compact import CompactCache
from services.retrieval.core.service import RetrievalService
from services.retrieval.core.store import IndexStore
from tests.retrieval.helpers import (TOY_DOCS, FakeEncoder, FakeTexts, bm25_reference, project_dir, request, tokens_of,
                                     write_bm25, write_embeddings, write_tfidf)

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False


class CountingPreprocess:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return tokens_of(text)


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn is not installed")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self._ctx = project_dir()
        self.root = self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__, None, None, None)
        write_bm25(self.root, "toy", TOY_DOCS)
        write_tfidf(self.root, "toy", TOY_DOCS)
        write_embeddings(self.root, "toy", TOY_DOCS)
        self.encoder, self.texts, self.pre = FakeEncoder(), FakeTexts({k: v.upper() for k, v in TOY_DOCS.items()}), CountingPreprocess()
        self.make_service()

    def make_service(self, **settings):
        self.settings = RetrievalSettings(stale_check_seconds=0.0, **settings)
        store = IndexStore(cache=CompactCache(self.root / "cache"), stale_check_seconds=0.0)
        self.service = RetrievalService(self.settings, store, self.encoder, self.texts, preprocess=self.pre)
        self.addCleanup(self.service.close)

    def ids(self, outcome):
        return [r["doc_id"] for r in outcome.results]

    # ------------------------------------------------------------------------------------------ single model
    def test_each_model_finds_the_obviously_relevant_document(self):
        for model in ("bm25", "tfidf", "embedding"):
            out = self.service.retrieve(request(model=model, query_tokens=["wing", "flutter"], query_text="wing flutter"))
            self.assertIn(self.ids(out)[0], ("d3", "d7"), model)
            self.assertEqual(out.model, model)
            self.assertEqual(len(out.results), out.to_response()["total"])

    def test_bm25_scores_match_the_reference(self):
        out = self.service.retrieve(request(model="bm25", query_tokens=["boundary", "layer"], top_k=3))
        reference = bm25_reference(TOY_DOCS, ["boundary", "layer"])
        for item in out.results:
            self.assertAlmostEqual(item["score"], reference[item["doc_id"]], places=12)
        self.assertEqual(len(out.results), 3)

    def test_results_are_plain_json_types(self):
        for model in ("bm25", "tfidf", "embedding"):
            item = self.service.retrieve(request(model=model, query_tokens=["flow"], query_text="flow")).results[0]
            self.assertIs(type(item["doc_id"]), str)
            self.assertIs(type(item["score"]), float)

    def test_response_shape(self):
        body = self.service.retrieve(request(query_tokens=["flow"])).to_response()
        self.assertEqual(set(body), {"results", "total", "model", "took_ms"})
        self.assertNotIn("text", body["results"][0])

    def test_include_text_attaches_texts(self):
        out = self.service.retrieve(request(query_tokens=["flutter"], include_text=True))
        self.assertTrue(all(r["text"] == TOY_DOCS[r["doc_id"]].upper() for r in out.results))
        self.assertEqual(len(self.texts.calls), 1)

    def test_database_failure_degrades_to_empty_texts_and_is_reported(self):
        self.texts.fail = True
        out = self.service.retrieve(request(query_tokens=["flutter"], include_text=True))
        self.assertTrue(out.results and all(r["text"] == "" for r in out.results))
        self.assertEqual(len(out.warnings), 1)
        self.assertIn("warnings", out.to_response())

    def test_no_text_lookup_when_nothing_was_found_or_not_requested(self):
        self.service.retrieve(request(query_tokens=["nothingmatches"], include_text=True))
        self.service.retrieve(request(query_tokens=["flow"], include_text=False))
        self.assertEqual(self.texts.calls, [])

    def test_unknown_words_give_an_empty_result_not_an_error(self):
        for model in ("bm25", "tfidf"):
            self.assertEqual(self.service.retrieve(request(model=model, query_tokens=["qwertyuiop"])).results, [])

    # ------------------------------------------------------------------------------------------ query policy
    def test_tokens_from_the_caller_are_not_preprocessed_again(self):
        self.service.retrieve(request(model="bm25", query_tokens=["flutter"], query_text="Flutter!"))
        self.service.retrieve(request(model="tfidf", query_tokens=["flutter"], query_text="Flutter!"))
        self.assertEqual(self.pre.calls, [])

    def test_text_only_request_works_for_every_model(self):
        """This is the RAG service's request: query_tokens is empty."""
        for model in ("bm25", "tfidf", "embedding"):
            out = self.service.retrieve(request(model=model, query_tokens=[], query_text="wing flutter model"))
            self.assertTrue(out.results, model)

    def test_empty_query_is_a_bad_request(self):
        for model in ("bm25", "tfidf", "embedding"):
            with self.assertRaises(InvalidQueryError):
                self.service.retrieve(request(model=model, query_tokens=[], query_text=""))

    def test_bad_query_is_reported_before_a_missing_index(self):
        with self.assertRaises(InvalidQueryError):
            self.service.retrieve(request(dataset="missing", query_tokens=[], query_text=""))

    def test_embedding_query_mode(self):
        self.service.retrieve(request(model="embedding", query_tokens=["wing", "flutter"], query_text="Wing flutter?"))
        self.assertEqual(self.encoder.calls[-1], "wing flutter")
        self.make_service(embedding_query_mode="raw")
        self.service.retrieve(request(model="embedding", query_tokens=["wing", "flutter"], query_text="Wing flutter?"))
        self.assertEqual(self.encoder.calls[-1], "Wing flutter?")

    def test_none_bm25_parameters_fall_back_to_the_defaults(self):
        a = self.service.retrieve(request(query_tokens=["boundary"], bm25_k1=None, bm25_b=None))
        b = self.service.retrieve(request(query_tokens=["boundary"], bm25_k1=1.5, bm25_b=0.75))
        self.assertEqual(a.results, b.results)

    def test_dataset_and_model_validation(self):
        with self.assertRaises(InvalidQueryError):
            self.service.retrieve(request(dataset=None, query_tokens=["a"]))
        with self.assertRaises(InvalidQueryError):
            self.service.retrieve(request(dataset="  ", query_tokens=["a"]))
        with self.assertRaises(InvalidQueryError):
            self.service.retrieve(request(model="splade", query_tokens=["a"]))
        with self.assertRaises(IndexNotFoundError):
            self.service.retrieve(request(dataset="nope", query_tokens=["a"]))

    def test_the_request_is_not_modified(self):
        req = request(model="bm25", query_tokens=["flow"])
        self.service.retrieve(req, model="tfidf")
        self.assertEqual(req.model, "bm25")

    # ------------------------------------------------------------------------------------------ freshness
    def test_a_rebuilt_index_is_noticed_without_a_restart(self):
        before = self.service.retrieve(request(query_tokens=["flutter"]))
        self.assertTrue(before.results)
        write_bm25(self.root, "toy", {"n1": "completely different words only", "n2": "nothing about flutter here at all folks"})
        after = self.service.retrieve(request(query_tokens=["flutter"]))
        self.assertEqual(self.ids(after), ["n2"])

    # ------------------------------------------------------------------------------------------ hybrid serial
    def test_serial_rescores_candidates_with_the_second_model(self):
        tokens = ["boundary", "layer", "flat", "plate"]
        out = self.service.hybrid_serial(request(first_model="bm25", second_model="tfidf", query_tokens=tokens, query_text="x", final_top_k=3))
        self.assertEqual(out.model, "hybrid_serial_bm25_tfidf")
        stage1 = self.service.retrieve(request(model="bm25", query_tokens=tokens, top_k=100))
        tfidf = {r["doc_id"]: r["score"] for r in self.service.retrieve(request(model="tfidf", query_tokens=tokens, top_k=100)).results}
        expected = sorted(((tfidf.get(r["doc_id"], 0.0), r["doc_id"]) for r in stage1.results), key=lambda p: -p[0])[:3]
        self.assertEqual([round(r["score"], 12) for r in out.results], [round(s, 12) for s, _ in expected])
        scores = [r["score"] for r in out.results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_serial_with_every_model_pair(self):
        for first in ("bm25", "tfidf", "embedding"):
            for second in ("bm25", "tfidf", "embedding"):
                out = self.service.hybrid_serial(request(first_model=first, second_model=second, query_tokens=["wing", "flutter"],
                                                         query_text="wing flutter", first_stage_k=10, final_top_k=4))
                self.assertTrue(out.results, (first, second))
                self.assertLessEqual(len(out.results), 4)

    def test_serial_same_model_keeps_the_stage_one_order(self):
        tokens = ["boundary", "layer"]
        out = self.service.hybrid_serial(request(first_model="bm25", second_model="bm25", query_tokens=tokens, query_text="x", final_top_k=3))
        direct = self.service.retrieve(request(model="bm25", query_tokens=tokens, top_k=3))
        self.assertEqual(out.results, direct.results)

    def test_serial_documents_missing_from_the_second_index_score_zero(self):
        write_tfidf(self.root, "toy", {k: v for k, v in TOY_DOCS.items() if k != "d4"})
        out = self.service.hybrid_serial(request(first_model="bm25", second_model="tfidf", query_tokens=["shock", "wave"],
                                                 query_text="x", final_top_k=5))
        scores = {r["doc_id"]: r["score"] for r in out.results}
        self.assertEqual(scores["d4"], 0.0)

    def test_serial_without_candidates_returns_nothing(self):
        out = self.service.hybrid_serial(request(first_model="bm25", second_model="tfidf", query_tokens=["qwertyuiop"], query_text="x"))
        self.assertEqual(out.results, [])

    # ------------------------------------------------------------------------------------------ hybrid parallel
    def test_parallel_fuses_with_the_ranking_package(self):
        from services.ranking.rankers import fuse_results

        tokens = ["boundary", "layer"]
        out = self.service.hybrid_parallel(request(models=["bm25", "tfidf"], query_tokens=tokens, query_text="x", top_k=4))
        per_model = [self.service.retrieve(request(model=m, query_tokens=tokens, top_k=100)).results for m in ("bm25", "tfidf")]
        expected = fuse_results(results_list=per_model, method="rrf", weights=None, rrf_k=60)[:4]
        self.assertEqual([r["doc_id"] for r in out.results], [r["doc_id"] for r in expected])
        self.assertEqual(out.model, "hybrid_parallel_bm25_tfidf_rrf")
        self.assertEqual(out.warnings, [])

    def test_parallel_weighted_fusion_passes_the_weights_in_model_order(self):
        seen = {}

        def fake_fuse(results_list, method, weights, rrf_k):
            seen.update(method=method, weights=weights, rrf_k=rrf_k, sizes=[len(r) for r in results_list])
            return results_list[0]

        self.service._fuse = fake_fuse
        self.service.hybrid_parallel(request(models=["tfidf", "bm25"], fusion_method="weighted", model_weights={"bm25": 0.7, "tfidf": 0.3},
                                             query_tokens=["flow"], query_text="x", rrf_k=40))
        self.assertEqual((seen["method"], seen["weights"], seen["rrf_k"]), ("weighted", [0.3, 0.7], 40))

    def test_parallel_reports_a_failing_model_instead_of_hiding_it(self):
        out = self.service.hybrid_parallel(request(models=["bm25", "tfidf", "embedding"], query_tokens=["wing"], query_text="wing", top_k=3))
        self.assertEqual(out.warnings, [])
        (self.root / "data" / "indices" / "toy" / "embedding" / "embeddings.npy").unlink()
        degraded = self.service.hybrid_parallel(request(models=["bm25", "tfidf", "embedding"], query_tokens=["wing"], query_text="wing", top_k=3))
        self.assertTrue(degraded.results)
        self.assertTrue(any(w.startswith("embedding failed") for w in degraded.warnings))
        self.assertIn("remaining models", degraded.warnings[-1])

    def test_parallel_raises_the_real_error_when_every_model_fails(self):
        with self.assertRaises(IndexNotFoundError):
            self.service.hybrid_parallel(request(dataset="nope", query_tokens=["wing"], query_text="wing"))

    def test_parallel_fusion_errors_become_retrieval_errors(self):
        def broken(*args):
            raise ValueError("bad weights")

        self.service._fuse = broken
        with self.assertRaises(RetrievalError):
            self.service.hybrid_parallel(request(query_tokens=["wing"], query_text="wing"))

    def test_parallel_needs_two_models_and_a_query(self):
        with self.assertRaises(InvalidQueryError):
            self.service.hybrid_parallel(request(models=["bm25"], query_tokens=["a"]))
        with self.assertRaises(InvalidQueryError):
            self.service.hybrid_parallel(request(query_tokens=[], query_text=""))

    def test_parallel_runs_the_preprocessing_once_for_a_text_only_query(self):
        self.service.hybrid_parallel(request(models=["bm25", "tfidf"], query_tokens=[], query_text="wing flutter"))
        self.assertEqual(self.pre.calls, ["wing flutter"])

    # ------------------------------------------------------------------------------------------ status
    def test_overall_status_lists_what_is_built(self):
        status = self.service.overall_status()
        self.assertEqual(status["service"], "retrieval")
        self.assertEqual(status["models"], {"tfidf": "ready", "embedding": "ready", "bm25": "ready"})
        self.assertEqual(status["datasets"]["toy"]["bm25"], "ready")
        (self.root / "data" / "indices" / "toy" / "bm25" / "bm25_index.pkl").unlink()
        self.assertEqual(self.service.overall_status("toy")["models"]["bm25"], "not_built")
        self.assertEqual(self.service.overall_status("unknown")["models"], {"tfidf": "not_built", "embedding": "not_built", "bm25": "not_built"})

    def test_model_status(self):
        bm25 = self.service.model_status("toy", "bm25")
        self.assertEqual((bm25["status"], bm25["dataset"], bm25["total_docs"]), ("ready", "toy", len(TOY_DOCS)))
        self.assertIn("avg_doc_length", bm25)
        tfidf = self.service.model_status("toy", "tfidf")
        self.assertEqual(tfidf["vocab_size"] > 0, True)
        embedding = self.service.model_status("toy", "embedding")
        self.assertEqual((embedding["embedding_dimension"], embedding["device"]), (32, "cpu"))
        for kind in ("bm25", "tfidf", "embedding"):
            missing = self.service.model_status("nope", kind)
            self.assertEqual(missing["status"], "not_built")
            self.assertIn(f"/index/{kind}/build", missing["message"])
        with self.assertRaises(InvalidQueryError):
            self.service.model_status(None, "bm25")

    # ------------------------------------------------------------------------------------------ warm-up
    def test_warmup_loads_the_requested_indexes_and_ignores_unbuilt_ones(self):
        self.service.warm(["toy:bm25|tfidf", "toy:embedding", "nothing", "toy:bogus"])
        loaded = {(i["dataset"], i["model"]) for i in self.service._store.loaded()}
        self.assertEqual(loaded, {("toy", "bm25"), ("toy", "tfidf"), ("toy", "embedding")})


if __name__ == "__main__":
    unittest.main()
