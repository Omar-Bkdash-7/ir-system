"""The gateway's search path, tested without FastAPI, HTTP or a dataset on disk.

These are the invariants that used to break in production:

* A query made only of stop words ("who are you") must still reach retrieval: the
  embedding model embeds ``query_text`` and never reads ``query_tokens``, so an
  empty result produced here hid the query from every model.
* The dataset's saved preprocessing config is read straight from the metadata
  file, never through ``services.indexing`` (which imports ``ir_datasets`` on
  every query).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from services.api_gateway.config import GatewaySettings
from services.api_gateway.core.dataset import dataset_info, fallback_info
from services.api_gateway.core.search import SearchService

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeClient:
    def __init__(self, tokens=("alpha",), retrieval=None, refinement=None):
        self.calls = []
        self.tokens = list(tokens)
        self.retrieval = retrieval if retrieval is not None else {
            "results": [{"doc_id": "d1", "score": 1.0, "text": "text"}], "total": 1,
        }
        self.refinement = refinement if refinement is not None else {
            "refined_query": "better query", "applied": True,
        }

    def post(self, url, payload, *, timeout, service):
        self.calls.append({"url": url, "payload": payload, "service": service, "timeout": timeout})
        if url.endswith("/preprocess"):
            return {"tokens": self.tokens}
        if url.endswith("/refine"):
            return dict(self.refinement)
        return dict(self.retrieval)

    def close(self):
        pass


def build(tokens=("alpha",), metadata_dir=None, **client_kwargs):
    client = FakeClient(tokens, **client_kwargs)
    settings = GatewaySettings(metadata_dir=str(metadata_dir) if metadata_dir else "data/nowhere")
    return SearchService(settings, client), client


def request(**kwargs):
    defaults = {"query": "heat transfer", "dataset": "cranfield", "model": "tfidf", "top_k": 20,
                "bm25_k1": 1.5, "bm25_b": 0.75, "enable_refinement": False, "query_history": []}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class StopWordQueryTest(unittest.TestCase):
    def test_a_query_without_tokens_still_reaches_the_embedding_model(self):
        service, client = build(tokens=())

        result = service.search_basic(request(model="embedding", dataset="cranfield"))

        retrieval = client.calls[-1]
        self.assertTrue(retrieval["url"].endswith("/retrieve"))
        self.assertEqual(retrieval["payload"]["query_tokens"], [])
        self.assertEqual(retrieval["payload"]["query_text"], "heat transfer")
        self.assertEqual(result["total"], 1)

    def test_the_hybrid_paths_are_not_short_circuited_either(self):
        service, client = build(tokens=())

        service.search_hybrid_parallel(request(models=["tfidf", "bm25"]))

        call = client.calls[-1]
        self.assertTrue(call["url"].endswith("/retrieve/hybrid/parallel"))
        self.assertEqual(call["payload"]["query_text"], "heat transfer")


class PayloadTest(unittest.TestCase):
    def test_basic_search_sends_the_model_and_cutoff(self):
        service, client = build()

        service.search_basic(request(model="bm25", top_k=7, bm25_k1=1.9, bm25_b=0.4))

        payload = client.calls[-1]["payload"]
        self.assertEqual(payload["model"], "bm25")
        self.assertEqual(payload["top_k"], 7)
        self.assertEqual(payload["bm25_k1"], 1.9)
        self.assertEqual(payload["bm25_b"], 0.4)
        self.assertTrue(payload["include_text"])
        self.assertEqual(payload["query_tokens"], ["alpha"])

    def test_the_dataset_is_only_sent_when_one_was_chosen(self):
        service, client = build()

        service.search_basic(request(dataset=None))
        self.assertNotIn("dataset", client.calls[-1]["payload"])

        service.search_basic(request(dataset="cranfield"))
        self.assertEqual(client.calls[-1]["payload"]["dataset"], "cranfield")

    def test_hybrid_serial_sends_both_stages(self):
        service, client = build()

        service.search_hybrid_serial(request(first_model="bm25", second_model="embedding",
                                             first_stage_k=500, final_top_k=10))

        call = client.calls[-1]
        self.assertTrue(call["url"].endswith("/retrieve/hybrid/serial"))
        self.assertEqual(call["payload"]["first_model"], "bm25")
        self.assertEqual(call["payload"]["second_model"], "embedding")
        self.assertEqual(call["payload"]["first_stage_k"], 500)
        self.assertEqual(call["payload"]["final_top_k"], 10)

    def test_hybrid_parallel_sends_the_fusion_parameters(self):
        service, client = build()

        service.search_hybrid_parallel(request(models=["tfidf", "bm25"], fusion_method="weighted",
                                               model_weights={"tfidf": 0.5, "bm25": 0.5},
                                               candidates_per_model=200, rrf_k=30))

        payload = client.calls[-1]["payload"]
        self.assertEqual(payload["models"], ["tfidf", "bm25"])
        self.assertEqual(payload["fusion_method"], "weighted")
        self.assertEqual(payload["model_weights"], {"tfidf": 0.5, "bm25": 0.5})
        self.assertEqual(payload["candidates_per_model"], 200)
        self.assertEqual(payload["rrf_k"], 30)


class RefinementTest(unittest.TestCase):
    def test_refinement_runs_first_and_its_text_is_what_gets_retrieved(self):
        service, client = build(refinement={"refined_query": "heat transfer coefficient", "applied": True})

        result = service.search_basic(request(enable_refinement=True, query_history=["heat"]))

        self.assertTrue(client.calls[0]["url"].endswith("/refine"))
        self.assertEqual(client.calls[0]["payload"]["history"], ["heat"])
        self.assertEqual(client.calls[-1]["payload"]["query_text"], "heat transfer coefficient")
        self.assertEqual(result["refinement"]["refined_query"], "heat transfer coefficient")

    def test_without_refinement_the_retrieval_call_comes_first_and_no_banner_is_added(self):
        service, client = build()

        result = service.search_basic(request())

        self.assertEqual(len(client.calls), 2)  # preprocess + retrieve only
        self.assertTrue(client.calls[0]["url"].endswith("/preprocess"))
        self.assertNotIn("refinement", result)


class ValidationTest(unittest.TestCase):
    def test_an_empty_query_is_rejected_before_any_call(self):
        service, client = build()

        with self.assertRaises(ValueError):
            service.search_basic(request(query="   "))

        self.assertEqual(client.calls, [])


class DatasetConfigTest(unittest.TestCase):
    def test_the_saved_preprocessing_config_is_sent_with_the_query(self):
        with tempfile.TemporaryDirectory() as scratch:
            Path(scratch, "cranfield.json").write_text(
                json.dumps({"dataset_name": "cranfield", "preprocessing_config": {"use_stemming": True}}),
                encoding="utf-8",
            )
            service, client = build(metadata_dir=scratch)

            service.search_basic(request())

        preprocess = next(call for call in client.calls if call["url"].endswith("/preprocess"))
        self.assertEqual(preprocess["payload"]["config"], {"use_stemming": True})

    def test_the_newest_metadata_file_wins(self):
        with tempfile.TemporaryDirectory() as scratch:
            old = Path(scratch, "old.json")
            old.write_text(json.dumps({"dataset_name": "old"}), encoding="utf-8")
            os.utime(old, (0, 0))
            fresh = Path(scratch, "fresh.json")
            fresh.write_text(json.dumps({"dataset_name": "fresh"}), encoding="utf-8")

            self.assertEqual(dataset_info(scratch)["dataset_name"], "fresh")

    def test_a_missing_directory_answers_with_the_fallback(self):
        self.assertEqual(dataset_info("data/definitely-not-here"), fallback_info())


class NoIndexingImportTest(unittest.TestCase):
    def test_reading_the_config_does_not_import_the_indexing_service(self):
        with tempfile.TemporaryDirectory() as scratch:
            metadata = Path(scratch) / "data" / "metadata"
            metadata.mkdir(parents=True)
            (metadata / "nano-beir-quora.json").write_text(
                json.dumps({"dataset_name": "nano-beir/quora", "preprocessing_config": {"use_stemming": True}}),
                encoding="utf-8",
            )
            script = (
                "import json, sys\n"
                "from services.api_gateway.config import GatewaySettings\n"
                "from services.api_gateway.core.search import SearchService\n"
                "service = SearchService(GatewaySettings(metadata_dir='data/metadata'))\n"
                "config = service.preprocessing_config('nano-beir/quora')\n"
                "loaded = sorted(m for m in sys.modules "
                "if m.startswith('services.indexing') or m == 'ir_datasets')\n"
                "print(json.dumps({'config': config, 'loaded': loaded}))\n"
            )
            result = subprocess.run([sys.executable, "-c", script], cwd=scratch, capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, timeout=180)

        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["config"], {"use_stemming": True})
        self.assertEqual(body["loaded"], [], "reading the config pulled in the indexing service / ir_datasets")


if __name__ == "__main__":
    unittest.main()
