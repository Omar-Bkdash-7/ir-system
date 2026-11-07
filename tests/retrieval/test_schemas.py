import unittest

try:
    from pydantic import ValidationError

    from services.retrieval.schemas import (HybridParallelRequest, HybridSerialRequest, RetrievalRequest, RetrievalResponse,
                                            SearchHit)
    HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover
    HAVE_PYDANTIC = False


@unittest.skipUnless(HAVE_PYDANTIC, "pydantic is not installed")
class RequestSchemaTests(unittest.TestCase):
    """The request models are unchanged by the rewrite: these tests pin their contract."""

    def test_defaults(self):
        req = RetrievalRequest()
        self.assertEqual((req.model, req.top_k, req.include_text, req.query_tokens, req.query_text, req.dataset),
                         ("tfidf", 50, True, [], "", None))
        self.assertEqual((req.bm25_k1, req.bm25_b), (1.5, 0.75))

    def test_model_is_normalised_and_validated(self):
        self.assertEqual(RetrievalRequest(model="BM25").model, "bm25")
        with self.assertRaises(ValidationError):
            RetrievalRequest(model="splade")

    def test_limits(self):
        for field, value in (("top_k", 0), ("top_k", 1001), ("bm25_k1", 3.1), ("bm25_k1", -0.1), ("bm25_b", 1.1), ("bm25_b", -0.1)):
            with self.assertRaises(ValidationError, msg=f"{field}={value}"):
                RetrievalRequest(**{field: value})

    def test_hybrid_serial(self):
        HybridSerialRequest(query_text="q", first_model="bm25", second_model="embedding")
        with self.assertRaises(ValidationError):
            HybridSerialRequest(first_model="bm25", second_model="tfidf")  # query_text is required
        with self.assertRaises(ValidationError):
            HybridSerialRequest(query_text="q", first_model="bm25", second_model="tfidf", first_stage_k=9)
        with self.assertRaises(ValidationError):
            HybridSerialRequest(query_text="q", first_model="nope", second_model="tfidf")

    def test_hybrid_parallel(self):
        ok = HybridParallelRequest(query_text="q", models=["BM25", "Embedding"])
        self.assertEqual((ok.models, ok.fusion_method, ok.rrf_k, ok.candidates_per_model), (["bm25", "embedding"], "rrf", 60, 1000))
        for models in (["bm25"], ["bm25", "bm25"], ["bm25", "tfidf", "embedding", "bm25"], ["bm25", "nope"]):
            with self.assertRaises(ValidationError, msg=str(models)):
                HybridParallelRequest(query_text="q", models=models)
        with self.assertRaises(ValidationError):
            HybridParallelRequest(query_text="q", models=["bm25", "tfidf"], fusion_method="borda")

    def test_weighted_fusion_needs_complete_weights_that_sum_to_one(self):
        ok = HybridParallelRequest(query_text="q", models=["bm25", "tfidf"], fusion_method="weighted", model_weights={"bm25": 0.6, "tfidf": 0.4})
        self.assertEqual(ok.fusion_method, "weighted")
        for weights in (None, {"bm25": 1.0}, {"bm25": 0.9, "tfidf": 0.9}):
            with self.assertRaises(ValidationError, msg=str(weights)):
                HybridParallelRequest(query_text="q", models=["bm25", "tfidf"], fusion_method="weighted", model_weights=weights)


@unittest.skipUnless(HAVE_PYDANTIC, "pydantic is not installed")
class ResponseSchemaTests(unittest.TestCase):
    def test_optional_fields_are_omitted_when_empty(self):
        body = RetrievalResponse(results=[{"doc_id": "a", "score": 1.0}], total=1).model_dump(exclude_none=True)
        self.assertEqual(body, {"results": [{"doc_id": "a", "score": 1.0}], "total": 1, "model": "tfidf"})

    def test_numeric_ids_are_returned_as_text(self):
        self.assertEqual(SearchHit(doc_id=17, score=0.5).doc_id, "17")

    def test_text_and_extras_are_kept(self):
        body = RetrievalResponse(results=[{"doc_id": "a", "score": 1, "text": "hello"}], total=1, model="bm25",
                                 took_ms=1.5, warnings=["w"]).model_dump(exclude_none=True)
        self.assertEqual(body["results"][0]["text"], "hello")
        self.assertEqual((body["took_ms"], body["warnings"]), (1.5, ["w"]))


if __name__ == "__main__":
    unittest.main()
