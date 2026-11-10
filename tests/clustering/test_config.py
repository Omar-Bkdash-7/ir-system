"""Settings come from the environment first, then configs/config.yaml, then the code defaults."""
import os
import unittest
from unittest import mock

from services.clustering.config import HDBSCAN_METRICS, ClusteringSettings, get_settings, reset_settings_cache


class ClusteringSettingsTest(unittest.TestCase):
    def setUp(self):
        reset_settings_cache()
        self.addCleanup(reset_settings_cache)

    def test_defaults_are_usable(self):
        settings = ClusteringSettings()
        self.assertEqual(settings.metric, "euclidean")
        self.assertGreaterEqual(settings.min_cluster_size, 2)
        self.assertGreaterEqual(settings.min_samples, 1)
        self.assertGreater(settings.n_keywords, 0)
        self.assertIn(settings.metric, HDBSCAN_METRICS)

    def test_environment_wins_over_configuration(self):
        with mock.patch.dict(
            os.environ,
            {
                "CLUSTERING_MIN_CLUSTER_SIZE": "12",
                "CLUSTERING_MIN_SAMPLES": "4",
                "CLUSTERING_METRIC": "cosine",
                "CLUSTERING_DATA_DIR": "/tmp/somewhere/clusters",
            },
            clear=False,
        ):
            settings = get_settings()
        self.assertEqual(settings.min_cluster_size, 12)
        self.assertEqual(settings.min_samples, 4)
        self.assertEqual(settings.metric, "cosine")
        self.assertEqual(str(settings.data_dir), "/tmp/somewhere/clusters")

    def test_an_unknown_metric_is_ignored_instead_of_crashing(self):
        with mock.patch.dict(os.environ, {"CLUSTERING_METRIC": "definitely-not-a-metric"}, clear=False):
            settings = get_settings()
        self.assertIn(settings.metric, HDBSCAN_METRICS)

    def test_a_non_numeric_count_is_ignored(self):
        with mock.patch.dict(os.environ, {"CLUSTERING_MIN_CLUSTER_SIZE": "many"}, clear=False):
            settings = get_settings()
        self.assertGreaterEqual(settings.min_cluster_size, 2)

    def test_settings_are_cached_until_reset(self):
        first = get_settings()
        self.assertIs(first, get_settings())
        reset_settings_cache()
        self.assertIsNot(first, get_settings())
