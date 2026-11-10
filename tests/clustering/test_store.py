"""The manifest is the commit point: derived files belong to one run, and unreadable data never crashes a read."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.clustering.core import store
from services.clustering.core.topics import TopicExtraction
from services.clustering.schemas import ClusterParams, TopicInfo


class ClusterStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name) / "clusters"
        self.params = ClusterParams(min_cluster_size=5, min_samples=3, metric="euclidean", n_keywords=10)
        self.labels = np.array([0, 0, 1, -1], dtype=np.int64)
        self.doc_ids = ["d0", "d1", "d2", "d3"]
        self.extraction = TopicExtraction(
            topics={0: TopicInfo(topic_id=0, keywords=["wing", "flow"], representative_docs=["d0"]), 1: TopicInfo(topic_id=1, keywords=["shock"])},
            method="ctfidf",
        )

    def _save(self, run_id="run-a", extraction=None, projection=None, dataset="cranfield"):
        return store.save(
            self.data_dir,
            dataset,
            run_id=run_id,
            params=self.params,
            index_fingerprint="fp1",
            doc_ids=self.doc_ids,
            cluster_labels=self.labels,
            extraction=extraction or self.extraction,
            embeddings_2d=projection,
            warnings=["note"],
        )

    def test_roundtrip(self):
        saved = self._save(projection=np.zeros((4, 2), dtype=np.float32))
        self.assertEqual((saved.n_clusters, saved.n_noise, saved.n_documents), (2, 1, 4))

        loaded = store.load(self.data_dir, "cranfield")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.cluster_labels, [0, 0, 1, -1])
        self.assertEqual(loaded.doc_ids, self.doc_ids)
        self.assertEqual(loaded.params, self.params)
        self.assertEqual(loaded.topics[0].keywords, ["wing", "flow"])
        self.assertEqual(loaded.topics_method, "ctfidf")
        self.assertTrue(loaded.topics_available)
        self.assertEqual(loaded.warnings, ["note"])
        self.assertEqual(loaded.embeddings_2d.shape, (4, 2))

    def test_no_projection_and_no_topics_are_valid(self):
        self._save(extraction=TopicExtraction(topics={}, method="none"))
        loaded = store.load(self.data_dir, "cranfield")
        self.assertIsNone(loaded.embeddings_2d)
        self.assertEqual(loaded.topics, {})
        self.assertFalse(loaded.topics_available)
        self.assertEqual(loaded.topics_method, "none")

    def test_a_new_run_removes_the_previous_run_files(self):
        self._save(run_id="run-a", projection=np.zeros((4, 2), dtype=np.float32))
        self._save(run_id="run-b", projection=np.ones((4, 2), dtype=np.float32))

        directory = self.data_dir / "cranfield"
        names = sorted(path.name for path in directory.iterdir())
        self.assertIn("run-run-b-embeddings_2d.npy", names)
        self.assertIn("run-run-b-topics.json", names)
        self.assertFalse(any(name.startswith("run-run-a-") for name in names))

    def test_the_manifest_only_points_at_its_own_run(self):
        self._save(run_id="run-a", projection=np.zeros((4, 2), dtype=np.float32))
        manifest = json.loads((self.data_dir / "cranfield" / "clusters.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "run-a")
        self.assertEqual(manifest["schema_version"], store.SCHEMA_VERSION)
        self.assertEqual(manifest["pipeline_version"], store.PIPELINE_VERSION)
        self.assertEqual(store.load(self.data_dir, "cranfield").pipeline_version, store.PIPELINE_VERSION)

    def test_a_run_from_an_older_pipeline_does_not_match(self):
        saved = self._save()
        self.assertTrue(saved.matches(self.params, "fp1", store.PIPELINE_VERSION))
        self.assertFalse(saved.matches(self.params, "fp1", store.PIPELINE_VERSION + 1))
        self.assertFalse(saved.matches(self.params, "fp2", store.PIPELINE_VERSION))

    def test_a_missing_derived_file_is_tolerated(self):
        self._save(projection=np.zeros((4, 2), dtype=np.float32))
        (self.data_dir / "cranfield" / "run-run-a-embeddings_2d.npy").unlink()
        loaded = store.load(self.data_dir, "cranfield")
        self.assertIsNotNone(loaded)
        self.assertIsNone(loaded.embeddings_2d)
        self.assertEqual(loaded.cluster_labels, [0, 0, 1, -1])

    def test_an_unreadable_manifest_is_reported_as_not_clustered(self):
        self._save()
        (self.data_dir / "cranfield" / "clusters.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(store.load(self.data_dir, "cranfield"))

    def test_an_unknown_schema_version_is_not_misread(self):
        self._save()
        path = self.data_dir / "cranfield" / "clusters.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 999
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIsNone(store.load(self.data_dir, "cranfield"))

    def test_nothing_stored_means_not_clustered(self):
        self.assertIsNone(store.load(self.data_dir, "cranfield"))
        self.assertEqual(store.clustered_datasets(self.data_dir), [])

    def test_the_summary_answers_status_questions_without_the_labels(self):
        self._save()
        summary = store.read_summary(self.data_dir, "cranfield")
        self.assertEqual(summary.dataset, "cranfield")
        self.assertEqual(summary.n_documents, 4)
        self.assertEqual(summary.n_clusters, 2)
        self.assertEqual(summary.n_noise, 1)
        self.assertEqual(summary.params, self.params)
        self.assertEqual(summary.pipeline_version, store.PIPELINE_VERSION)

    def test_a_summary_is_rebuilt_when_only_the_manifest_exists(self):
        self._save()
        (self.data_dir / "cranfield" / "summary.json").unlink()
        summary = store.read_summary(self.data_dir, "cranfield")
        self.assertIsNotNone(summary)
        self.assertEqual(summary.n_clusters, 2)

    def test_datasets_are_listed_by_name_not_by_slug(self):
        self._save(dataset="beir/quora")
        self.assertEqual(store.clustered_datasets(self.data_dir), ["beir/quora"])

    def test_delete(self):
        self._save()
        self.assertTrue(store.delete(self.data_dir, "cranfield"))
        self.assertFalse(store.delete(self.data_dir, "cranfield"))
        self.assertIsNone(store.load(self.data_dir, "cranfield"))

    def test_labels_and_doc_ids_must_agree(self):
        with self.assertRaises(ValueError):
            store.save(
                self.data_dir,
                "cranfield",
                run_id="run-x",
                params=self.params,
                index_fingerprint="fp1",
                doc_ids=["only-one"],
                cluster_labels=self.labels,
                extraction=self.extraction,
            )
