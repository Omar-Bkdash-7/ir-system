"""Cache correctness, locking, graceful degradation and the read endpoints of the clustering service."""
import json
import tempfile
import unittest
from pathlib import Path

from services.clustering.errors import (
    ClusteringInProgressError,
    ClusteringUnavailableError,
    ClustersNotFoundError,
    DocumentNotClusteredError,
    EmbeddingIndexMissingError,
    InvalidClusteringParameterError,
    TextsUnavailableError,
)
from services.indexing.storage import try_lock
from tests.clustering.helpers import FakeTextLoader, make_harness, write_embedding_index


class ClusterDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.harness = make_harness(Path(self.tmp.name))

    def test_a_run_is_stored_and_described(self):
        response = self.harness.run()
        self.assertEqual(response.dataset, "cranfield")
        self.assertEqual(response.n_clusters, 2)
        self.assertEqual(response.n_noise, 0)
        self.assertEqual(response.n_documents, 10)
        self.assertEqual(response.params.min_cluster_size, 5)
        self.assertTrue(response.topics_available)
        self.assertEqual(response.topics_method, "ctfidf")
        self.assertEqual(len(response.cluster_labels), 10)
        self.assertEqual(len(response.embeddings_2d), 10)
        self.assertEqual([cluster.size for cluster in response.clusters], [5, 5])
        self.assertTrue((self.harness.settings.data_dir / "cranfield" / "clusters.json").exists())

    def test_the_cache_is_used_when_nothing_changed(self):
        service = self.harness.service()
        self.harness.run(service)
        second = self.harness.run(service)
        self.assertEqual(self.harness.clusterer.calls, 1)
        self.assertEqual(second.n_clusters, 2)

    def test_new_parameters_recompute(self):
        service = self.harness.service()
        self.harness.run(service)
        response = self.harness.run(service, min_cluster_size=7)
        self.assertEqual(self.harness.clusterer.calls, 2)
        self.assertEqual(response.params.min_cluster_size, 7)

    def test_a_rebuilt_embedding_index_recomputes(self):
        service = self.harness.service()
        self.harness.run(service)
        self.harness.rewrite_index()
        self.harness.run(service)
        self.assertEqual(self.harness.clusterer.calls, 2)

    def test_a_run_from_an_older_pipeline_version_is_recomputed(self):
        service = self.harness.service()
        self.harness.run(service)
        manifest_path = self.harness.settings.data_dir / "cranfield" / "clusters.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["pipeline_version"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        self.harness.run(service)
        self.assertEqual(self.harness.clusterer.calls, 2)
        self.assertFalse(self.service_stale_check(service))

    def service_stale_check(self, service):
        return service.get_status("cranfield").stale

    def test_force_recompute_always_runs(self):
        service = self.harness.service()
        self.harness.run(service, force_recompute=True)
        self.harness.run(service, force_recompute=True)
        self.assertEqual(self.harness.clusterer.calls, 2)

    def test_a_missing_embedding_index_is_a_typed_error(self):
        harness = make_harness(Path(self.tmp.name) / "no-index", with_index=False)
        with self.assertRaises(EmbeddingIndexMissingError):
            harness.run()

    def test_parameters_larger_than_the_dataset_are_refused_before_clustering(self):
        with self.assertRaises(InvalidClusteringParameterError):
            self.harness.run(min_cluster_size=50)
        self.assertEqual(self.harness.clusterer.calls, 0)

    def test_missing_libraries_are_a_service_unavailable_error(self):
        def unavailable():
            raise ClusteringUnavailableError("hdbscan is not installed")

        service = self.harness.service(availability_check=unavailable)
        with self.assertRaises(ClusteringUnavailableError):
            self.harness.run(service)
        self.assertEqual(self.harness.clusterer.calls, 0)

    def test_a_second_concurrent_run_for_the_same_dataset_is_refused(self):
        with try_lock("clustering:cranfield") as acquired:
            self.assertTrue(acquired, "the test must hold the lock")
            with self.assertRaises(ClusteringInProgressError):
                self.harness.run()
        # ...and the lock is released afterwards, so the next run succeeds.
        self.harness.run()
        self.assertEqual(self.harness.clusterer.calls, 1)

    def test_the_projection_can_be_skipped(self):
        response = self.harness.run(include_visualization=False)
        self.assertEqual(self.harness.projector.calls, 0)
        self.assertIsNone(response.embeddings_2d)

    def test_a_dataset_id_with_a_slash_gets_a_flat_directory(self):
        dataset = "beir/quora"
        write_embedding_index(
            self.harness.settings.indices_dir, dataset, self.harness.embeddings, self.harness.doc_ids
        )
        self.harness.dataset = dataset
        response = self.harness.run()
        self.assertEqual(response.dataset, dataset)
        self.assertTrue((self.harness.settings.data_dir / "beir_quora" / "clusters.json").exists())
        self.assertFalse((self.harness.settings.data_dir / "beir").exists())


class GracefulTopicsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.harness = make_harness(Path(self.tmp.name))

    def test_without_a_database_the_clusters_are_kept_and_topics_are_off(self):
        self.harness.text_loader.error = TextsUnavailableError("connection refused")
        response = self.harness.run()

        self.assertEqual(response.n_clusters, 2)
        self.assertFalse(response.topics_available)
        self.assertEqual(response.topics_method, "none")
        self.assertEqual(response.clusters[0].topic, None)
        self.assertTrue(any("topics unavailable" in warning for warning in response.warnings))

    def test_placeholder_texts_are_never_stored(self):
        self.harness.text_loader.error = TextsUnavailableError("no table")
        self.harness.run()

        manifest = json.loads(
            (self.harness.settings.data_dir / "cranfield" / "clusters.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["topics_available"])
        self.assertFalse(list(self.harness.settings.data_dir.rglob("*topics.json")))

    def test_too_many_documents_without_text_disables_topics(self):
        self.harness.text_loader = FakeTextLoader(texts=[""] * 10, missing=10)
        response = self.harness.run()
        self.assertFalse(response.topics_available)
        self.assertTrue(any("have no text" in warning for warning in response.warnings))

    def test_a_few_missing_texts_only_warn(self):
        harness = make_harness(Path(self.tmp.name) / "partial", max_missing_text_ratio=0.5)
        harness.text_loader = FakeTextLoader(texts=harness.texts, missing=1)
        response = harness.run()
        self.assertTrue(response.topics_available)
        self.assertTrue(any("no text" in warning for warning in response.warnings))

    def test_a_failing_topic_extractor_does_not_lose_the_clusters(self):
        def boom(*args, **kwargs):
            raise RuntimeError("bertopic exploded")

        service = self.harness.service(topic_extractor=boom)
        response = self.harness.run(service)
        self.assertEqual(response.n_clusters, 2)
        self.assertFalse(response.topics_available)
        self.assertTrue(any("topics unavailable" in warning for warning in response.warnings))


class ReadEndpointsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.harness = make_harness(Path(self.tmp.name))
        self.service = self.harness.service()

    def test_a_document_reports_its_cluster_and_topic(self):
        self.harness.run(self.service)
        info = self.service.get_document_cluster("cranfield", "doc0")
        self.assertEqual(info.cluster_id, 0)
        self.assertIsNotNone(info.topic)
        self.assertEqual(info.topic.keywords, ["wing", "flow"])

    def test_an_unknown_document_is_its_own_error(self):
        self.harness.run(self.service)
        with self.assertRaises(DocumentNotClusteredError):
            self.service.get_document_cluster("cranfield", "not-a-document")

    def test_a_noise_document_has_no_topic(self):
        self.harness.clusterer.labels[0] = -1
        self.harness.run(self.service)
        info = self.service.get_document_cluster("cranfield", "doc0")
        self.assertEqual(info.cluster_id, -1)
        self.assertIsNone(info.topic)

    def test_reading_before_any_run_is_not_found(self):
        with self.assertRaises(ClustersNotFoundError):
            self.service.get_clusters("cranfield")
        with self.assertRaises(ClustersNotFoundError):
            self.service.get_document_cluster("cranfield", "doc0")

    def test_status_reports_state_parameters_and_staleness(self):
        self.assertEqual(self.service.get_status("cranfield").status, "not_clustered")

        self.harness.run(self.service)
        status = self.service.get_status("cranfield")
        self.assertEqual(status.status, "clustered")
        self.assertEqual(status.n_documents, 10)
        self.assertEqual(status.params.min_samples, 3)
        self.assertFalse(status.stale)

        self.harness.rewrite_index()
        self.assertTrue(self.service.get_status("cranfield").stale)

    def test_datasets_lists_stored_runs(self):
        self.assertEqual(self.service.list_clustered_datasets(), [])
        self.harness.run(self.service)
        self.assertEqual(self.service.list_clustered_datasets(), ["cranfield"])

    def test_delete(self):
        self.harness.run(self.service)
        self.service.delete_clusters("cranfield")
        self.assertEqual(self.service.list_clustered_datasets(), [])
        with self.assertRaises(ClustersNotFoundError):
            self.service.delete_clusters("cranfield")
