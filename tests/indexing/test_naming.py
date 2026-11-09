"""naming.py: the single place that turns dataset ids into slugs, table names and safe paths."""
import os
import tempfile
import unittest
from pathlib import Path

from services.indexing.errors import InvalidDatasetIdError, InvalidIndexTypeError
from services.indexing.naming import (
    PG_IDENTIFIER_LIMIT, dataset_id_parts, index_slug, safe_child_path, table_name, table_to_dataset_id,
    validate_index_type,
)


class IndexSlugTest(unittest.TestCase):
    def test_keeps_the_existing_on_disk_layout(self):
        # existing index directories were created with exactly these rules - they must never change
        self.assertEqual(index_slug("cranfield"), "cranfield")
        self.assertEqual(index_slug("beir/quora"), "beir_quora")
        self.assertEqual(index_slug("nano-beir/quora"), "nano_beir_quora")
        # A dot must not survive here: retrieval and common.config.get_dataset_specific_path build the same slug with
        # '.' -> '_', so a dotted dataset was built in a directory retrieval never looked in (the index was invisible).
        self.assertEqual(index_slug("demo/v1.2"), "demo_v1_2")

    def test_rejects_names_that_would_point_outside_the_indices_directory(self):
        for bad in ("", ".", ".."):
            with self.subTest(bad=bad), self.assertRaises(InvalidDatasetIdError):
                index_slug(bad)

    def test_invalid_names_are_value_errors(self):
        # common.exception_handlers turns ValueError into HTTP 400, so no handler had to be added
        with self.assertRaises(ValueError):
            index_slug("..")

    def test_index_types(self):
        for ok in ("tfidf", "bm25", "embedding"):
            self.assertEqual(validate_index_type(ok), ok)
        with self.assertRaisesRegex(InvalidIndexTypeError, "Invalid index type: nope"):
            validate_index_type("nope")


class TableNameTest(unittest.TestCase):
    def test_matches_what_common_database_builds_for_ordinary_ids(self):
        # common.database.get_documents_by_ids builds f"dataset_{id with '/' and '-' replaced by '_'}"
        for dataset_id in ("cranfield", "vaswani", "beir/quora", "nano-beir/quora", "msmarco-passage/trec-dl-2019"):
            with self.subTest(dataset_id=dataset_id):
                expected = "dataset_" + dataset_id.replace("/", "_").replace("-", "_")
                self.assertEqual(table_name(dataset_id, "dataset"), expected)

    def test_lower_cases_and_replaces_dots(self):
        # PostgreSQL folds unquoted names to lower case: the old code asked information_schema for the mixed-case
        # name and never found the table it had just created
        self.assertEqual(table_name("Demo/V1.2", "dataset"), "dataset_demo_v1_2")

    def test_refuses_ids_that_could_not_be_pasted_into_sql(self):
        for bad in ("x; drop table y", "a b", "a'b", 'a"b', "a%b", "x--y;", "tab\tle"):
            with self.subTest(bad=bad), self.assertRaises(InvalidDatasetIdError):
                table_name(bad, "dataset")


class DatasetIdPartsTest(unittest.TestCase):
    def test_normal_ids(self):
        self.assertEqual(dataset_id_parts("beir/quora"), ["beir", "quora"])
        self.assertEqual(dataset_id_parts("/beir/quora/"), ["beir", "quora"])

    def test_dangerous_ids(self):
        for bad in ("", "   ", "/", "..", ".", "../x", "a/../b", "a/./b", "x\x00y"):
            with self.subTest(bad=bad), self.assertRaises(InvalidDatasetIdError):
                dataset_id_parts(bad)


class SafeChildPathTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve() / "ir_home"
        self.root.mkdir()

    def test_inside_the_root(self):
        self.assertEqual(safe_child_path(self.root, "beir/quora"), self.root / "beir" / "quora")

    def test_never_the_root_itself_or_outside(self):
        for bad in ("", ".", "..", "../outside", "a/../.."):
            with self.subTest(bad=bad), self.assertRaises(InvalidDatasetIdError):
                safe_child_path(self.root, bad)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks not available")
    def test_a_symlink_cannot_lead_out_of_the_root(self):
        outside = self.root.parent / "outside"
        outside.mkdir()
        try:
            os.symlink(outside, self.root / "link", target_is_directory=True)
        except OSError:
            self.skipTest("cannot create symlinks here")
        with self.assertRaises(InvalidDatasetIdError):
            safe_child_path(self.root, "link/x")


class TableToDatasetIdTest(unittest.TestCase):
    def test_exact_id_from_the_metadata_wins(self):
        # 'dataset_nano_beir_quora' cannot tell 'nano-beir/quora' from 'nano/beir/quora' or 'nano_beir_quora'
        table = "dataset_nano_beir_quora"
        self.assertEqual(table_to_dataset_id(table, "dataset", ["nano-beir/quora"]), "nano-beir/quora")
        self.assertEqual(table_to_dataset_id(table, "dataset", []), "nano/beir/quora")  # old heuristic = fallback

    def test_postgres_truncated_names_are_still_recognised(self):
        long_id = "collection/" + "x" * 80
        truncated = table_name(long_id, "dataset")[:PG_IDENTIFIER_LIMIT]
        self.assertEqual(table_to_dataset_id(truncated, "dataset", [long_id]), long_id)

    def test_unusable_known_ids_are_ignored(self):
        self.assertEqual(table_to_dataset_id("dataset_beir_quora", "dataset", ["bad id;", "beir/quora"]), "beir/quora")


if __name__ == "__main__":
    unittest.main()
