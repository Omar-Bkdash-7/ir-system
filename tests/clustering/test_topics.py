"""Keywords come from real words, and representative documents are the ones closest to their cluster centroid."""
import re
import unittest

import numpy as np

from services.clustering.core.topics import extract_topics
from tests.clustering.helpers import two_cluster_embeddings

WORD = re.compile(r"^[^\W\d_]{3,}$")


class ExtractTopicsTest(unittest.TestCase):
    def setUp(self):
        self.embeddings, self.doc_ids, self.texts, self.labels = two_cluster_embeddings(per_cluster=4)

    def _extract(self, **kwargs):
        options = dict(n_keywords=5, representative_count=2)
        options.update(kwargs)
        return extract_topics(self.texts, self.embeddings, self.labels, **options)

    def test_keywords_are_words_not_numbers_or_empty_strings(self):
        result = self._extract()
        self.assertEqual(result.method, "ctfidf")
        self.assertEqual(sorted(result.topics), [0, 1])
        for topic in result.topics.values():
            self.assertTrue(topic.keywords, "every cluster should get keywords")
            for keyword in topic.keywords:
                self.assertTrue(WORD.match(keyword), f"unexpected keyword {keyword!r}")

    def test_noise_documents_get_no_topic(self):
        labels = self.labels.copy()
        labels[0] = -1
        result = extract_topics(self.texts, self.embeddings, labels, n_keywords=5)
        self.assertNotIn(-1, result.topics)

    def test_representatives_are_the_documents_closest_to_the_centroid(self):
        # Cluster 0: three identical (unit) vectors plus one far outlier whose text happens to be the longest.
        # The embedding index stores L2-normalised vectors, so this is the shape real data has.
        outlier = np.array([1.0, -1.0], dtype=np.float32) / np.sqrt(2.0)
        embeddings = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                outlier,
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 0, 0], dtype=np.int64)
        texts = [
            "wing aerodynamics lift drag",
            "wing aerodynamics lift drag",
            "wing aerodynamics lift drag",
            "wing aerodynamics lift drag " + "outlier " * 50,
        ]
        result = extract_topics(texts, embeddings, labels, n_keywords=5, representative_count=1)
        self.assertEqual(result.topics[0].representative_docs, ["wing aerodynamics lift drag"])

    def test_without_text_topics_are_reported_as_unavailable(self):
        result = extract_topics([""] * len(self.texts), self.embeddings, self.labels)
        self.assertEqual(result.method, "none")
        self.assertEqual(result.topics, {})
        self.assertFalse(result.available)

    def test_an_all_noise_run_has_no_topics(self):
        result = extract_topics(self.texts, self.embeddings, np.full(len(self.texts), -1))
        self.assertEqual(result.method, "none")

    def test_mismatched_inputs_are_a_programming_error(self):
        with self.assertRaises(ValueError):
            extract_topics(self.texts, self.embeddings, self.labels[:-1])
        with self.assertRaises(ValueError):
            extract_topics(self.texts[:-1], self.embeddings, self.labels)
