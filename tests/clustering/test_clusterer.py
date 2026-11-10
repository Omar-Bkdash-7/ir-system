"""Parameter validation happens before HDBSCAN runs, and the projection never takes the result down with it."""
import unittest

import numpy as np

from services.clustering.core.clusterer import (
    HDBSCAN_AVAILABLE,
    project_2d,
    run_hdbscan,
    validate_parameters,
)
from services.clustering.errors import InvalidClusteringParameterError
from tests.clustering.helpers import two_cluster_embeddings


class ValidateParametersTest(unittest.TestCase):
    def test_acceptable_parameters_pass(self):
        validate_parameters(min_cluster_size=5, min_samples=3, metric="euclidean", n_keywords=10, n_docs=100)

    def test_a_cluster_smaller_than_two_is_refused(self):
        with self.assertRaises(InvalidClusteringParameterError):
            validate_parameters(min_cluster_size=1, min_samples=1, metric="euclidean", n_keywords=10)

    def test_min_samples_above_min_cluster_size_is_refused(self):
        with self.assertRaises(InvalidClusteringParameterError):
            validate_parameters(min_cluster_size=3, min_samples=5, metric="euclidean", n_keywords=10)

    def test_an_unknown_metric_is_refused(self):
        with self.assertRaises(InvalidClusteringParameterError) as raised:
            validate_parameters(min_cluster_size=5, min_samples=3, metric="levenshtein", n_keywords=10)
        self.assertIn("metric", str(raised.exception))

    def test_a_cluster_size_larger_than_the_dataset_is_refused(self):
        with self.assertRaises(InvalidClusteringParameterError):
            validate_parameters(min_cluster_size=50, min_samples=3, metric="euclidean", n_keywords=10, n_docs=10)


class ProjectionTest(unittest.TestCase):
    def test_tiny_datasets_use_the_deterministic_fallback(self):
        embeddings, _, _, _ = two_cluster_embeddings(per_cluster=2)
        first = project_2d(embeddings[:3])
        second = project_2d(embeddings[:3])
        self.assertEqual(first.shape, (3, 2))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.allclose(first, second))

    def test_a_single_document_still_gets_two_coordinates(self):
        embeddings, _, _, _ = two_cluster_embeddings(per_cluster=1)
        self.assertEqual(project_2d(embeddings[:1]).shape, (1, 2))

    def test_one_row_per_document(self):
        embeddings, _, _, _ = two_cluster_embeddings(per_cluster=4)
        self.assertEqual(project_2d(embeddings).shape, (len(embeddings), 2))


@unittest.skipUnless(HDBSCAN_AVAILABLE, "hdbscan is not installed in this image")
class HdbscanTest(unittest.TestCase):
    def test_two_separated_blobs_are_two_clusters(self):
        embeddings, _, _, _ = two_cluster_embeddings(per_cluster=6, dimension=4)
        labels = run_hdbscan(embeddings, min_cluster_size=3, min_samples=2, metric="euclidean")
        self.assertEqual(len(labels), len(embeddings))
        self.assertEqual(len(set(labels.tolist()) - {-1}), 2)
