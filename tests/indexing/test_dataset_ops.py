"""Local dataset deletion, fix-corrupted, download/debug helpers and metadata lookups."""
import unittest
from unittest import mock

from services.indexing.dataset import maintenance
from services.indexing.dataset.deleter import LocalDatasetDeleter
from services.indexing.dataset.metadata_manager import DatasetMetadataManager, known_dataset_ids
from tests.indexing._support import FakeDoc, FakeIrDataset, WorkdirTestCase


class FakeIr:
    """Stands in for the ir_datasets module: knows some ids, raises KeyError for the others (like the real one)."""

    def __init__(self, **datasets):
        self.datasets, self.loaded = datasets, []

    def load(self, dataset_id):
        self.loaded.append(dataset_id)
        if dataset_id not in self.datasets:
            raise KeyError(dataset_id)
        return self.datasets[dataset_id]


class LocalDatasetDeleterTest(WorkdirTestCase):
    def setUp(self):
        super().setUp()
        (self.ir_home / "beir" / "quora").mkdir(parents=True)
        (self.ir_home / "beir" / "quora" / "docs.tsv").write_text("x")
        (self.ir_home / "beir" / "other").mkdir()
        (self.ir_home / "cranfield").mkdir()
        (self.work / "precious.txt").write_text("keep me")  # lives NEXT TO the datasets directory

    def test_deletes_only_the_requested_dataset_and_its_metadata(self):
        DatasetMetadataManager().save_metadata({"doc_count": 1}, "beir/quora")
        result = LocalDatasetDeleter().delete_dataset("beir/quora")
        self.assertEqual((result["status"], result["deleted"], result["metadata_cleaned"]), ("success", "beir/quora", True))
        self.assertFalse((self.ir_home / "beir" / "quora").exists())
        self.assertTrue((self.ir_home / "beir" / "other").exists())
        self.assertTrue((self.ir_home / "cranfield").exists())
        self.assertFalse(DatasetMetadataManager().metadata_exists("beir/quora"))

    def test_unknown_dataset(self):
        with self.assertRaises(FileNotFoundError):
            LocalDatasetDeleter().delete_dataset("nope/nothing")

    def test_ids_that_would_leave_the_datasets_directory_delete_nothing(self):
        # The old code resolved ".." to the PARENT of the datasets directory and "/" to the datasets directory
        # itself - and then deleted it with shutil.rmtree.
        for bad in ("", "   ", "/", ".", "..", "../x", "beir/../..", "beir/../../"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                LocalDatasetDeleter().delete_dataset(bad)
        self.assertTrue(self.ir_home.exists())
        self.assertTrue((self.work / "precious.txt").exists())
        self.assertTrue((self.ir_home / "beir" / "quora" / "docs.tsv").exists())
        self.assertTrue((self.ir_home / "cranfield").exists())


class FixCorruptedDatasetTest(WorkdirTestCase):
    def setUp(self):
        super().setUp()
        (self.ir_home / "beir" / "quora").mkdir(parents=True)
        (self.ir_home / "beir" / "quora" / "broken.tsv").write_text("corrupt")
        (self.work / "precious.txt").write_text("keep me")

    def _run(self, dataset_id, fake_ir):
        with mock.patch.object(maintenance, "ir_datasets", fake_ir):
            return maintenance.fix_corrupted_dataset(dataset_id)

    def test_deletes_the_cached_files_and_downloads_again(self):
        docs = [FakeDoc(f"d{i}", f"text {i}") for i in range(12)]
        fake = FakeIr(**{"beir/quora": FakeIrDataset(docs, queries=["q1", "q2"], qrels=["r1"])})
        result = self._run("beir/quora", fake)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["counts"], {"documents": 12, "queries": 2, "qrels": 1})
        self.assertFalse((self.ir_home / "beir" / "quora" / "broken.tsv").exists())

    def test_an_unknown_id_fails_BEFORE_anything_is_deleted(self):
        # the original handler deleted the files first, then failed (NameError: ir_datasets was never imported)
        with self.assertRaises(KeyError):
            self._run("beir/quora-typo", FakeIr())
        self.assertTrue((self.ir_home / "beir" / "quora" / "broken.tsv").exists())

    def test_unsafe_ids_delete_nothing(self):
        for bad in ("", "  ", "/", "..", "../x", "beir/../.."):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._run(bad, FakeIr())
        self.assertTrue((self.work / "precious.txt").exists())
        self.assertTrue((self.ir_home / "beir" / "quora" / "broken.tsv").exists())


class MaintenanceHelpersTest(WorkdirTestCase):
    def test_download_only_samples_the_first_documents(self):
        docs = [FakeDoc(f"d{i}", "t") for i in range(50)]
        fake = FakeIr(demo=FakeIrDataset(docs, queries=["q"] * 3, qrels=["r"] * 4))
        with mock.patch.object(maintenance, "ir_datasets", fake):
            result = maintenance.download_dataset("demo")
        self.assertEqual(result["sample_counts"], {"docs_sampled": 11, "queries": 3, "qrels": 4})

class KnownDatasetIdsTest(WorkdirTestCase):
    def test_exact_ids_come_from_the_metadata_files(self):
        manager = DatasetMetadataManager()
        manager.save_metadata({}, "nano-beir/quora")
        manager.save_metadata({}, "cranfield")
        self.assertEqual(known_dataset_ids(), ["cranfield", "nano-beir/quora"])

    def test_never_raises(self):
        with mock.patch("services.indexing.dataset.metadata_manager.list_all_datasets", side_effect=OSError("boom")):
            self.assertEqual(known_dataset_ids(), [])
        self.assertEqual(known_dataset_ids(), [])  # no metadata directory at all


class AtomicMetadataWriteTest(WorkdirTestCase):
    def test_a_rewrite_replaces_the_file_and_leaves_no_temporary_file(self):
        manager = DatasetMetadataManager()
        manager.save_metadata({"doc_count": 1}, "demo")
        manager.save_metadata({"doc_count": 2}, "demo")
        files = sorted(p.name for p in (self.work / "data" / "metadata").iterdir())
        self.assertEqual(files, ["demo.json"], "a temporary file survived the write")
        self.assertEqual(manager.load_metadata("demo")["doc_count"], 2)


if __name__ == "__main__":
    unittest.main()
