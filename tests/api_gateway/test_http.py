"""HTTP level: the gateway must forward the status a downstream service answered with.

Every downstream failure used to be swallowed by the generic handler and reported as a 500, so a missing index
(a plain 404 from the retrieval service) looked like a server bug.
"""
import importlib.util
import logging
import os
import tempfile
import unittest
from unittest import mock

import httpx

GATEWAY_DOWNSTREAM_ERRORS = ("search_basic", "search_hybrid_serial", "search_hybrid_parallel")


def have(*modules):
    return all(importlib.util.find_spec(module) is not None for module in modules)


def downstream_error(status_code, detail):
    request = httpx.Request("POST", "http://retrieval:8003/retrieve")
    response = httpx.Response(status_code, json={"error": "IndexNotFound", "detail": detail}, request=request)
    return httpx.HTTPStatusError(f"{status_code} from retrieval", request=request, response=response)


@unittest.skipUnless(have("fastapi", "httpx"), "FastAPI test client (fastapi + httpx) is not installed")
class GatewayHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        previous = os.getcwd()
        # importing main calls setup_logging(), which creates logs/api_gateway.log relative to the working directory
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
            os.chdir(scratch)
            try:
                from services.api_gateway.main import app
            finally:
                os.chdir(previous)
        cls.client = TestClient(app, raise_server_exceptions=False)
        logging.disable(logging.CRITICAL)
        cls.addClassCleanup(logging.disable, logging.NOTSET)

    def test_a_missing_index_is_a_404_not_a_500(self):
        with mock.patch("services.api_gateway.main.search_basic",
                        side_effect=downstream_error(404, "TF-IDF index not found at data/indices/toy")):
            response = self.client.post("/search", json={"query": "heat transfer", "dataset": "toy"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "TF-IDF index not found at data/indices/toy")

    def test_a_rejected_downstream_request_keeps_its_status(self):
        with mock.patch("services.api_gateway.main.search_hybrid_parallel",
                        side_effect=downstream_error(422, "Model must be one of ['tfidf', 'embedding', 'bm25']")):
            response = self.client.post("/search/hybrid/parallel",
                                        json={"query": "heat", "models": ["bm25", "nope"]})
        self.assertEqual(response.status_code, 422)
        self.assertIn("Model must be one of", response.json()["detail"])

    def test_an_unexpected_failure_is_still_a_500(self):
        for name in GATEWAY_DOWNSTREAM_ERRORS:
            with self.subTest(downstream=name), mock.patch(f"services.api_gateway.main.{name}",
                                                           side_effect=RuntimeError("boom")):
                payload = {"query": "heat", "models": ["bm25", "tfidf"]} if name == "search_hybrid_parallel" else {"query": "heat"}
                if name == "search_hybrid_serial":
                    payload.update(first_model="bm25", second_model="tfidf")
                path = {  # the route each client function backs
                    "search_basic": "/search",
                    "search_hybrid_serial": "/search/hybrid/serial",
                    "search_hybrid_parallel": "/search/hybrid/parallel",
                }[name]
                self.assertEqual(self.client.post(path, json=payload).status_code, 500)

    def test_an_empty_query_is_a_400(self):
        self.assertEqual(self.client.post("/search", json={"query": "   "}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
