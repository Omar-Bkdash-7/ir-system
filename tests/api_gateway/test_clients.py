"""The gateway's search calls must survive a query that preprocessing empties, and must not load the indexing service.

A query made only of stop words ("who are you") used to be answered with an empty result by the gateway itself, so
even the embedding model - which embeds query_text and never reads query_tokens - could not answer it. The saved
preprocessing config, in turn, used to be read through services.indexing, which imports ir_datasets on every query.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.api_gateway.clients import retrieval_client

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


class _FakeClient:
    """Stands in for httpx.Client inside retrieval_client."""

    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, json=None):
        _FakeClient.calls.append((url, json))
        return _FakeResponse({"results": [{"doc_id": "d1", "score": 1.0, "text": "text"}], "total": 1,
                              "model": "embedding"})


class StopWordQueryTest(unittest.TestCase):
    def search(self, call, **kwargs):
        _FakeClient.calls = []
        with mock.patch.object(retrieval_client, "_preprocess_query", return_value=[]), \
                mock.patch.object(retrieval_client.httpx, "Client", _FakeClient):
            result = call("who are you", **kwargs)
        self.assertEqual(len(_FakeClient.calls), 1, "the retrieval service was never called")
        return _FakeClient.calls[0], result

    def test_a_query_without_tokens_still_reaches_the_embedding_model(self):
        (url, payload), result = self.search(retrieval_client.search_basic, model="embedding", dataset="cranfield")
        self.assertTrue(url.endswith("/retrieve"))
        self.assertEqual(payload["query_tokens"], [])
        self.assertEqual(payload["query_text"], "who are you")
        self.assertEqual(result["total"], 1)

    def test_the_hybrid_calls_are_not_short_circuited_either(self):
        ((url, payload), _) = self.search(retrieval_client.search_hybrid_parallel, models=["tfidf", "bm25"])
        self.assertTrue(url.endswith("/retrieve/hybrid/parallel"))
        self.assertEqual(payload["query_text"], "who are you")


class NoIndexingImportTest(unittest.TestCase):
    def test_the_preprocessing_config_is_read_without_importing_indexing(self):
        with tempfile.TemporaryDirectory() as scratch:
            metadata = Path(scratch) / "data" / "metadata"
            metadata.mkdir(parents=True)
            (metadata / "nano-beir-quora.json").write_text(
                json.dumps({"dataset_name": "nano-beir/quora", "preprocessing_config": {"use_stemming": True}}),
                encoding="utf-8")
            script = (
                "import json, sys\n"
                "from services.api_gateway.clients.retrieval_client import _get_preprocessing_config_for_dataset as f\n"
                "loaded = sorted(m for m in sys.modules "
                "if m.startswith('services.indexing') or m == 'ir_datasets')\n"
                "print(json.dumps({'config': f('nano-beir/quora'), 'loaded': loaded}))\n"
            )
            result = subprocess.run([sys.executable, "-c", script], cwd=scratch, capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["config"], {"use_stemming": True})
        self.assertEqual(body["loaded"], [], "loading the config pulled in the indexing service / ir_datasets")


if __name__ == "__main__":
    unittest.main()
