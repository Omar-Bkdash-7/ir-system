"""End-to-end over HTTP, on an indexed dataset that already exists: preprocessing -> retrieval -> fusion -> ranking.

These tests cross the unit boundaries on purpose: they check what a user of the system sees, including the
consistency between two places that run the same code (the retrieval service fuses in-process with
services.ranking.rankers, the ranking service fuses behind /fuse).
"""
import unittest

from tests.system import _stack
from tests.system._stack import call

MODELS = ("bm25", "tfidf", "embedding")


def setUpModule():
    _stack.require_stack()


class PipelineTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        response = call("preprocessing", "/preprocess", {"text": _stack.QUERY})
        if response.status != 200 or not response.json().get("tokens"):
            raise unittest.SkipTest(f"preprocessing cannot tokenize the query: {response!r}")
        cls.tokens = response.json()["tokens"]

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def retrieve(self, model, top_k=10):
        return call("retrieval", "/retrieve", {"query_tokens": self.tokens, "query_text": _stack.QUERY,
                                               "model": model, "top_k": top_k, "include_text": True,
                                               "dataset": _stack.DATASET})

    def fuse(self, method, models=MODELS, top_k=5, candidates=100, weights=None, rrf_k=60):
        payload = {"query_tokens": self.tokens, "query_text": _stack.QUERY, "models": list(models),
                   "fusion_method": method, "top_k": top_k, "candidates_per_model": candidates,
                   "dataset": _stack.DATASET, "rrf_k": rrf_k}
        if weights:
            payload["model_weights"] = weights
        return call("retrieval", "/retrieve/hybrid/parallel", payload)

    def search(self, model="bm25", top_k=5):
        return call("api_gateway", "/search", {"query": _stack.QUERY, "model": model, "top_k": top_k,
                                               "dataset": _stack.DATASET})

    def search_hybrid(self, models=("bm25", "tfidf"), method="rrf", top_k=5, candidates=100, rrf_k=60):
        return call("api_gateway", "/search/hybrid/parallel", {
            "query": _stack.QUERY, "models": list(models), "fusion_method": method, "top_k": top_k,
            "candidates_per_model": candidates, "rrf_k": rrf_k, "dataset": _stack.DATASET})

    @staticmethod
    def doc_ids(response):
        return [hit["doc_id"] for hit in response.json()["results"]]

    @staticmethod
    def assert_ordered_by_score(case, hits):
        scores = [hit["score"] for hit in hits]
        case.assertEqual(scores, sorted(scores, reverse=True), "results are not ordered by score")

    # ---- preprocessing ----------------------------------------------------------------------------------------------
    def test_preprocessing_turns_the_query_into_indexable_tokens(self):
        self.assertTrue(self.tokens, "no tokens")
        self.assertEqual(self.tokens, [token.lower() for token in self.tokens], "tokens are not lowercased")
        self.assertTrue(all(token.isalnum() for token in self.tokens), "punctuation survived preprocessing")
        for stopword in ("what", "when", "of"):
            self.assertNotIn(stopword, self.tokens, "stopwords were not removed")
        self.assertIn("law", self.tokens, "lemmatization did not run ('laws' -> 'law')")

    # ---- retrieval --------------------------------------------------------------------------------------------------
    def test_every_retriever_serves_the_indexed_dataset(self):
        for model in MODELS:
            with self.subTest(model=model):
                response = self.retrieve(model)
                self.assertEqual(response.status, 200, response.raw[:300])
                body = response.json()
                hits = body["results"]
                self.assertEqual(body["total"], len(hits))
                self.assertGreaterEqual(len(hits), 5)
                self.assert_ordered_by_score(self, hits)
                self.assertIn(_stack.RELEVANT_DOC, [hit["doc_id"] for hit in hits[:5]],
                              f"{model} lost the known relevant document")
                self.assertTrue(hits[0].get("text"), "include_text=True returned no text")

    def test_fusion_methods_agree_and_do_not_duplicate(self):
        orders = {}
        for method in ("rrf", "combsum", "combmnz"):
            with self.subTest(method=method):
                response = self.fuse(method)
                self.assertEqual(response.status, 200, response.raw[:300])
                doc_ids = self.doc_ids(response)
                self.assertEqual(len(doc_ids), 5)
                self.assertEqual(len(set(doc_ids)), len(doc_ids), f"{method} returned the same document twice")
                self.assertIn(_stack.RELEVANT_DOC, doc_ids)
                orders[method] = doc_ids
        self.assertEqual(orders["rrf"], orders["combsum"])
        self.assertEqual(orders["rrf"], orders["combmnz"])

    def test_weighted_fusion_needs_weights_that_sum_to_one(self):
        good = self.fuse("weighted", weights={"bm25": 0.5, "tfidf": 0.3, "embedding": 0.2})
        self.assertEqual(good.status, 200, good.raw[:300])
        self.assertIn(_stack.RELEVANT_DOC, self.doc_ids(good))
        bad = self.fuse("weighted", weights={"bm25": 2.0, "tfidf": 1.0, "embedding": 0.5})
        self.assertEqual(bad.status, 422)
        self.assertIn("sum to 1.0", bad.raw)

    def test_fusion_is_deterministic(self):
        first, second = self.fuse("rrf"), self.fuse("rrf")
        self.assertEqual(self.doc_ids(first), self.doc_ids(second))

    # ---- gateway: the same chain over HTTP ---------------------------------------------------------------------------
    def test_the_gateway_basic_search_matches_the_retrieval_service(self):
        direct = self.retrieve("bm25", top_k=5)
        through_gateway = self.search("bm25", top_k=5)
        self.assertEqual(through_gateway.status, 200, through_gateway.raw[:300])
        self.assertEqual(self.doc_ids(through_gateway), self.doc_ids(direct),
                         "the gateway found different documents than a direct retrieval call")

    def test_the_gateway_hybrid_parallel_matches_the_retrieval_fusion(self):
        direct = self.fuse("rrf", models=("bm25", "tfidf"), top_k=5, candidates=100)
        through_gateway = self.search_hybrid(models=("bm25", "tfidf"), top_k=5, candidates=100)
        self.assertEqual(through_gateway.status, 200, through_gateway.raw[:300])
        self.assertEqual(self.doc_ids(through_gateway), self.doc_ids(direct))

    # ---- ranking on real retrieval output ----------------------------------------------------------------------------
    def test_ranking_orders_the_fused_output_and_keeps_every_field(self):
        fused = self.search_hybrid(top_k=5)
        self.assertEqual(fused.status, 200, fused.raw[:300])
        hits = fused.json()["results"]
        response = call("ranking", "/rank", {"results": hits, "method": "score", "top_k": 3})
        self.assertEqual(response.status, 200, response.raw[:300])
        body = response.json()
        self.assertEqual([hit["doc_id"] for hit in body["ranked_results"]], [hit["doc_id"] for hit in hits[:3]])
        self.assertEqual(body["ranked_results"][0]["text"], hits[0]["text"], "ranking dropped the text field")
        self.assertIn("took_ms", body)
        self.assertEqual(body["total"], len(body["ranked_results"]))

    def test_the_ranking_service_fuses_exactly_like_the_retrieval_service(self):
        lists = []
        for model in ("bm25", "tfidf"):
            response = call("retrieval", "/retrieve", {
                "query_tokens": self.tokens, "query_text": _stack.QUERY, "model": model, "top_k": 20,
                "include_text": False, "dataset": _stack.DATASET})
            self.assertEqual(response.status, 200, response.raw[:300])
            lists.append([{"doc_id": hit["doc_id"], "score": hit["score"]} for hit in response.json()["results"]])

        behind_the_endpoint = call("ranking", "/fuse", {"result_lists": lists, "method": "rrf",
                                                       "rrf_k": 60, "top_k": 5})
        self.assertEqual(behind_the_endpoint.status, 200, behind_the_endpoint.raw[:300])
        fused_by_retrieval = self.fuse("rrf", models=("bm25", "tfidf"), top_k=5, candidates=20)
        self.assertEqual(fused_by_retrieval.status, 200, fused_by_retrieval.raw[:300])
        self.assertEqual([hit["doc_id"] for hit in behind_the_endpoint.json()["fused_results"]],
                         self.doc_ids(fused_by_retrieval),
                         "the ranking service and the retrieval service fused the same lists differently")

    # ---- error contracts ----------------------------------------------------------------------------------------------
    def test_client_errors_are_reported_instead_of_hidden(self):
        cases = [
            ("ranking top_k below 1", "ranking", "/rank", {"results": [{"score": 1.0}], "top_k": 0}, 422),
            ("ranking non-numeric score", "ranking", "/rank", {"results": [{"doc_id": "x", "score": "high"}]}, 400),
            ("ranking single list", "ranking", "/fuse", {"result_lists": [[{"doc_id": "x", "score": 1.0}]]}, 422),
            ("retrieval unknown model", "retrieval", "/retrieve", {"query_tokens": ["heat"], "model": "nope"}, 422),
            ("gateway empty query", "api_gateway", "/search", {"query": "   "}, 400),
            ("indexing invalid preprocessing config", "indexing", "/index/tfidf/build",
             {"preprocessing_config": {"bogus_key": 1}}, 400),
        ]
        for label, service, path, body, expected in cases:
            with self.subTest(case=label):
                params = {"dataset_name": "zz-test/does-not-exist"} if service == "indexing" else None
                response = call(service, path, body, params=params)
                self.assertEqual(response.status, expected, f"{label}: {response.raw[:300]}")

    def test_a_dataset_without_an_index_is_a_404_all_the_way_through(self):
        response = call("api_gateway", "/search", {"query": "heat transfer", "model": "bm25",
                                                   "dataset": "zz-test/does-not-exist"})
        self.assertEqual(response.status, 404,
                         f"a missing index must stay a 404 (not a 500): {response.raw[:300]}")

    def test_building_an_index_for_a_dataset_that_was_never_uploaded_is_a_400(self):
        response = call("indexing", "/index/bm25/build", method="POST",
                        params={"dataset_name": "zz-test/does-not-exist"})
        self.assertEqual(response.status, 400, response.raw[:300])
        self.assertIn("not uploaded to the database", response.raw)


if __name__ == "__main__":
    unittest.main()
