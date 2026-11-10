"""Dataset ids map to exactly the directory the indexing service wrote, with no way to escape the data root."""
import unittest
from pathlib import Path

from services.clustering.core.paths import dataset_dir, dataset_slug, embedding_files, run_artifact
from services.clustering.errors import InvalidClusteringParameterError


class DatasetSlugTest(unittest.TestCase):
    def test_slashes_dashes_and_dots_become_underscores(self):
        self.assertEqual(dataset_slug("beir/quora"), "beir_quora")
        self.assertEqual(dataset_slug("ms-marco"), "ms_marco")
        self.assertEqual(dataset_slug("trec.covid"), "trec_covid")

    def test_the_indexing_rule_is_reused(self):
        from services.indexing.naming import index_slug

        for dataset in ("beir/quora", "cranfield", "a-b.c/d"):
            self.assertEqual(dataset_slug(dataset), index_slug(dataset))

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(dataset_slug("  cranfield  "), "cranfield")

    def test_an_unusable_id_is_a_client_error(self):
        for bad in ("", "   ", "/", "../..", "///"):
            with self.assertRaises(InvalidClusteringParameterError):
                dataset_slug(bad)

    def test_the_directory_is_flat(self):
        data_dir = Path("/data/clusters")
        self.assertEqual(dataset_dir(data_dir, "beir/quora"), data_dir / "beir_quora")

    def test_embedding_files_follow_the_index_layout(self):
        npy, doc_map = embedding_files(Path("/data/indices"), "beir/quora")
        self.assertEqual(npy, Path("/data/indices/beir_quora/embedding/embeddings.npy"))
        self.assertEqual(doc_map, Path("/data/indices/beir_quora/embedding/doc_id_map.json"))

    def test_run_artifacts_are_namespaced_by_run(self):
        self.assertEqual(
            run_artifact(Path("/data/clusters/cranfield"), "abc", "topics.json").name,
            "run-abc-topics.json",
        )
