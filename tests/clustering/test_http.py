"""HTTP contract: status codes for every outcome, and dataset ids that contain '/' stay routable."""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from services.clustering.main import create_app
from services.indexing.storage import try_lock
from tests.clustering.helpers import make_harness, write_embedding_index


class ClusteringApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.harness = make_harness(Path(self.tmp.name))
        self.service = self.harness.service()
        self.client = TestClient(create_app(self.service))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def _cluster(self, **overrides):
        payload = {"dataset": "cranfield", **overrides}
        return self.client.post("/cluster", json=payload)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("clustering_available", response.json())

    def test_cluster_returns_the_documented_shape(self):
        response = self._cluster()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in ("dataset", "n_clusters", "n_noise", "n_documents", "clusters", "cluster_labels", "params"):
            self.assertIn(key, body)
        self.assertEqual(body["params"], {"min_cluster_size": 5, "min_samples": 3, "metric": "euclidean", "n_keywords": 10})
        self.assertEqual(body["clusters"][0]["size"], 5)

    def test_validation_errors_are_422(self):
        self.assertEqual(self._cluster(min_cluster_size=1).status_code, 422)
        self.assertEqual(self._cluster(metric="levenshtein").status_code, 422)
        self.assertEqual(self._cluster(dataset="   ").status_code, 422)

    def test_a_parameter_that_cannot_work_is_400_with_a_message(self):
        response = self._cluster(min_cluster_size=50)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "InvalidClusteringParameterError")
        self.assertIn("larger than the dataset", response.json()["detail"])

    def test_clusters_are_readable_and_can_skip_the_projection(self):
        self._cluster()
        full = self.client.get("/clusters/cranfield")
        self.assertEqual(full.status_code, 200)
        self.assertEqual(len(full.json()["embeddings_2d"]), 10)

        light = self.client.get("/clusters/cranfield", params={"include_visualization": "false"})
        self.assertEqual(light.status_code, 200)
        self.assertIsNone(light.json()["embeddings_2d"])

    def test_unknown_dataset_is_404(self):
        response = self.client.get("/clusters/nothing-here")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "ClustersNotFoundError")

    def test_status_and_datasets(self):
        self.assertEqual(self.client.get("/status/cranfield").json()["status"], "not_clustered")
        self.assertEqual(self.client.get("/datasets").json()["datasets"], [])

        self._cluster()
        status = self.client.get("/status/cranfield").json()
        self.assertEqual(status["status"], "clustered")
        self.assertEqual(status["n_documents"], 10)
        self.assertFalse(status["stale"])
        self.assertEqual(self.client.get("/datasets").json()["datasets"], ["cranfield"])

    def test_document_lookup(self):
        self._cluster()
        response = self.client.get("/document/cranfield/doc0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cluster_id"], 0)

        missing = self.client.get("/document/cranfield/unknown")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"], "DocumentNotClusteredError")

    def test_a_dataset_id_with_a_slash_is_reachable(self):
        dataset = "beir/quora"
        write_embedding_index(
            self.harness.settings.indices_dir, dataset, self.harness.embeddings, self.harness.doc_ids
        )
        self.assertEqual(self._cluster(dataset=dataset).status_code, 200)
        self.assertEqual(self.client.get("/clusters/beir/quora").status_code, 200)
        self.assertEqual(self.client.get("/status/beir/quora").json()["status"], "clustered")
        self.assertEqual(self.client.get("/document/beir/quora/doc1").status_code, 200)

    def test_a_running_clustering_is_409(self):
        with try_lock("clustering:cranfield") as acquired:
            self.assertTrue(acquired)
            response = self._cluster()
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "ClusteringInProgressError")

    def test_delete_then_read_is_404(self):
        self._cluster()
        self.assertEqual(self.client.delete("/clusters/cranfield").status_code, 200)
        self.assertEqual(self.client.delete("/clusters/cranfield").status_code, 404)

    def test_a_missing_embedding_index_is_404(self):
        harness = make_harness(Path(self.tmp.name) / "no-index", with_index=False)
        client = TestClient(create_app(harness.service()))
        response = client.post("/cluster", json={"dataset": "cranfield"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "EmbeddingIndexMissingError")
