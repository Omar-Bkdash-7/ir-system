import unittest

try:
    from pydantic import ValidationError

    from services.ranking.schemas import FusionRequest, FusionResponse, Hit, RankingRequest, RankingResponse
    HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover
    HAVE_PYDANTIC = False

LISTS = [[{"doc_id": "a", "score": 1.0}], [{"doc_id": "b", "score": 2.0}]]


@unittest.skipUnless(HAVE_PYDANTIC, "pydantic is not installed")
class RankingRequestTests(unittest.TestCase):
    def test_defaults_are_the_old_ones(self):
        req = RankingRequest(results=[{"score": 1}])
        self.assertEqual((req.method, req.top_k), ("cosine", 50))

    def test_results_are_required_and_keep_every_field(self):
        with self.assertRaises(ValidationError):
            RankingRequest()
        item = {"doc_id": "a", "score": 1, "text": None, "metadata": {"x": [1, 2]}}
        self.assertEqual(RankingRequest(results=[item]).results, [item])

    def test_top_k_bounds(self):
        for value in (0, -3, 10_001):
            with self.assertRaises(ValidationError, msg=str(value)):
                RankingRequest(results=[], top_k=value)
        self.assertEqual(RankingRequest(results=[], top_k=10_000).top_k, 10_000)

    def test_response_shape(self):
        body = RankingResponse(ranked_results=[{"score": 1, "x": 2}], total=1).model_dump()
        self.assertEqual(body, {"ranked_results": [{"score": 1, "x": 2}], "total": 1, "took_ms": None})


@unittest.skipUnless(HAVE_PYDANTIC, "pydantic is not installed")
class FusionRequestTests(unittest.TestCase):
    def test_defaults(self):
        req = FusionRequest(result_lists=LISTS)
        self.assertEqual((req.method, req.rrf_k, req.top_k, req.weights), ("rrf", 60, 50, None))

    def test_between_two_and_ten_lists(self):
        with self.assertRaises(ValidationError):
            FusionRequest(result_lists=LISTS[:1])
        with self.assertRaises(ValidationError):
            FusionRequest(result_lists=LISTS * 6)
        self.assertEqual(len(FusionRequest(result_lists=LISTS * 5).result_lists), 10)

    def test_method_is_validated_and_lower_cased(self):
        self.assertEqual(FusionRequest(result_lists=LISTS, method="CombMNZ").method, "combmnz")
        with self.assertRaises(ValidationError):
            FusionRequest(result_lists=LISTS, method="borda")

    def test_weighted_needs_one_valid_weight_per_list(self):
        ok = FusionRequest(result_lists=LISTS, method="weighted", weights=[3, 1])
        self.assertEqual(ok.weights, [3.0, 1.0])
        for weights in (None, [], [1.0], [1.0, 1.0, 1.0], [-1.0, 2.0], [0.0, 0.0]):
            with self.assertRaises(ValidationError, msg=str(weights)):
                FusionRequest(result_lists=LISTS, method="weighted", weights=weights)
        self.assertEqual(FusionRequest(result_lists=LISTS, method="rrf", weights=[9, 9, 9]).method, "rrf")  # ignored

    def test_numeric_bounds(self):
        for field, value in (("rrf_k", 0), ("rrf_k", 101), ("top_k", 0), ("top_k", 10_001)):
            with self.assertRaises(ValidationError, msg=f"{field}={value}"):
                FusionRequest(result_lists=LISTS, **{field: value})

    def test_hits(self):
        self.assertEqual(Hit(doc_id=17, score=1).doc_id, "17")
        with self.assertRaises(ValidationError):
            Hit(doc_id="a", score=float("nan"))
        with self.assertRaises(ValidationError):
            Hit(doc_id="a", score=float("inf"))
        with self.assertRaises(ValidationError):
            Hit(doc_id="a")

    def test_response_shape(self):
        body = FusionResponse(fused_results=[{"doc_id": "a", "score": 0.5}], total=1, method="rrf").model_dump()
        self.assertEqual(body, {"fused_results": [{"doc_id": "a", "score": 0.5}], "total": 1, "method": "rrf", "took_ms": None})


if __name__ == "__main__":
    unittest.main()
