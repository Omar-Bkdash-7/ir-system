"""The index build must preprocess the documents with the configuration the request asked for.

Regression: `preprocess(text, config)` passed the config into the reserved `mode` parameter (the second positional),
so every index was built with DEFAULT processing while the response still said `preprocessing_applied: true`.
"""
import contextlib
import json
import sys
import types
import unittest
from unittest import mock

from tests.indexing._support import FakeDatabase, WorkdirTestCase, have

DATASET = "demo/ds"
TABLE = "dataset_demo_ds"
VALID_KEYS = {"use_stemming", "remove_stopwords", "min_token_length"}


@contextlib.contextmanager
def patched_modules(replacements):
    """Publish `replacements` in sys.modules for the duration of the block, then remove exactly those keys.

    mock.patch.dict(sys.modules, ...) restores the WHOLE dict, so every module imported inside the block (torch,
    numpy, ...) disappears again and the next import of it blows up. Only the keys we added may be undone.
    """
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def recording_preprocessing(calls):
    """sys.modules entries replacing services.preprocessing, recording exactly how preprocess() is called."""
    module = types.ModuleType("services.preprocessing.preprocessing_service")

    class PreprocessingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def from_dict(cls, data):
            unknown = set(data) - VALID_KEYS
            if unknown:
                raise ValueError(f"Invalid configuration keys: {unknown}")
            return cls(**data)

        def to_dict(self):
            return dict(self.kwargs)

    def preprocess(text, mode=None, config=None):
        calls.append({"text": text, "mode": mode, "config": config})
        return [word.lower() for word in text.split() if word.isalpha()]

    module.preprocess, module.PreprocessingConfig = preprocess, PreprocessingConfig
    return {"services.preprocessing": types.ModuleType("services.preprocessing"),
            "services.preprocessing.preprocessing_service": module}


class PreprocessingConfigTest(WorkdirTestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.enterContext(patched_modules(recording_preprocessing(self.calls)))
        database = FakeDatabase(tables=[TABLE], rows={TABLE: [("d1", "running dogs"), ("d2", "quickly running")]})
        connection = mock.patch("common.database.get_db_connection", database.connect)
        connection.start()
        self.addCleanup(connection.stop)

    def build(self, config, name="out.jsonl"):
        from services.indexing.services.build_service import _preprocess_documents
        path = self.work / name
        _preprocess_documents(DATASET, config, str(path))
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_the_config_reaches_the_preprocessor_and_not_the_mode_parameter(self):
        documents = self.build({"use_stemming": True, "remove_stopwords": True})
        self.assertEqual([call["text"] for call in self.calls], ["running dogs", "quickly running"])
        self.assertEqual([call["mode"] for call in self.calls], [None, None],
                         "the config leaked into the unused `mode` parameter, so default processing was used")
        self.assertEqual([call["config"].kwargs for call in self.calls],
                         [{"use_stemming": True, "remove_stopwords": True}] * 2)
        self.assertEqual([document["doc_id"] for document in documents], ["d1", "d2"])

    def test_a_config_object_is_accepted_as_well(self):
        from services.preprocessing.preprocessing_service import PreprocessingConfig
        config = PreprocessingConfig(remove_stopwords=False)
        self.build(config, "object.jsonl")
        self.assertEqual([call["config"] for call in self.calls], [config, config])

    def test_the_default_config_applies_when_none_is_given(self):
        self.build(None, "default.jsonl")
        self.assertEqual([call["config"].kwargs for call in self.calls], [{}, {}])

    def test_an_invalid_config_is_a_client_error_not_a_broken_build(self):
        from services.indexing.services.build_service import InvalidPreprocessingConfigError
        with self.assertRaises(InvalidPreprocessingConfigError) as caught:
            self.build({"bogus_key": 1})
        self.assertEqual(caught.exception.reason, "invalid_preprocessing_config")
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(self.calls, [], "nothing may be preprocessed with an invalid config")


class RememberedConfigTest(unittest.TestCase):
    """A build with a custom config must record it: the query side preprocesses with the saved config."""

    def setUp(self):
        from services.indexing.services import build_service

        self.build_service = build_service
        self.saved = []
        recorded = self.saved

        class FakeBuilder:
            def build(self, *args, **kwargs):
                return {"status": "success", "vocab_size": 3}

        class FakeDatasetService:
            def has_metadata(self, dataset_name):
                return False

            def save_preprocessing_config(self, dataset_name, config):
                recorded.append((dataset_name, config))

        builder = mock.patch.object(self.build_service, "TFIDFIndexBuilder", FakeBuilder)
        service = mock.patch("services.indexing.dataset.service.DatasetService", FakeDatasetService)
        for patcher in (builder, service):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_successful_build_with_a_config_saves_it(self):
        result = self.build_service.build_tfidf_index("demo/ds", False, {"use_stemming": True})
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.saved, [("demo/ds", {"use_stemming": True})])

    def test_a_default_build_saves_nothing(self):
        self.build_service.build_tfidf_index("demo/ds", False, None)
        self.assertEqual(self.saved, [])

    def test_a_failed_build_saves_nothing(self):
        failure = mock.patch.object(self.build_service.TFIDFIndexBuilder, "build",
                                   lambda *a, **k: {"status": "error", "reason": "build_failed"})
        failure.start()
        self.addCleanup(failure.stop)
        self.build_service.build_tfidf_index("demo/ds", False, {"use_stemming": True})
        self.assertEqual(self.saved, [])


class DefaultRebuildTest(unittest.TestCase):
    """A rebuild with the defaults must clear an older custom config.

    Leaving it behind makes the query side keep preprocessing with a configuration the rebuilt index no longer has.
    """

    def test_a_default_rebuild_clears_the_stored_config(self):
        from services.indexing.services import build_service

        saved = []

        class FakeBuilder:
            def build(self, *args, **kwargs):
                return {"status": "success", "vocab_size": 3}

        class FakeDatasetService:
            def has_metadata(self, dataset_name):
                return True

            def save_preprocessing_config(self, dataset_name, config):
                saved.append((dataset_name, config))

        with mock.patch.object(build_service, "TFIDFIndexBuilder", FakeBuilder), \
                mock.patch("services.indexing.dataset.service.DatasetService", FakeDatasetService):
            build_service.build_tfidf_index("demo/ds", False, None)
        self.assertEqual(saved, [("demo/ds", None)])

    def test_build_all_does_not_record_a_config_when_every_index_was_skipped(self):
        from services.indexing.services import build_service

        saved = []

        class Existing:
            """Exists on disk (so the shared preprocessing pass is skipped) and refuses to rebuild."""

            def __init__(self, dataset_name=None):
                pass

            def exists(self):
                return True

            def build(self, *args, **kwargs):
                return {"status": "skipped", "message": "index already exists"}

        class FakeDatasetService:
            def has_metadata(self, dataset_name):
                return True

            def save_preprocessing_config(self, dataset_name, config):
                saved.append((dataset_name, config))

        patchers = [mock.patch.object(build_service, name, Existing) for name in
                    ("TFIDFIndexer", "BM25Indexer", "EmbeddingIndexer",
                     "TFIDFIndexBuilder", "BM25IndexBuilder", "EmbeddingIndexBuilder")]
        patchers += [mock.patch.object(build_service, "_check_dataset_in_database", lambda *a, **k: None),
                     mock.patch("services.indexing.dataset.service.DatasetService", FakeDatasetService)]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        result = build_service.build_all_indices("demo/ds", False, {"use_stemming": True})

        self.assertEqual(result["status"], "success")
        self.assertEqual(saved, [], "nothing was rebuilt, so the stored config must stay untouched")


class SavedConfigTest(WorkdirTestCase):
    """The config must be stored even for a dataset that has no metadata file yet (uploaded, never validated).

    Regression: save_preprocessing_config() needed an existing metadata file, so a build with a custom config kept
    it nowhere and api_gateway preprocessed the queries with the defaults — a different vocabulary than the index.
    """

    DATASET = "beir/nfcorpus"
    CONFIG = {"use_stemming": True, "remove_stopwords": True}

    def service(self):
        from services.indexing.dataset.service import DatasetService
        return DatasetService()

    def test_a_dataset_without_metadata_gets_a_metadata_file(self):
        from services.indexing.dataset.metadata_manager import known_dataset_ids
        service = self.service()
        self.assertFalse(service.has_metadata(self.DATASET))

        service.save_preprocessing_config(self.DATASET, self.CONFIG)

        self.assertTrue(service.has_metadata(self.DATASET))
        self.assertEqual(service.get_preprocessing_config(self.DATASET), self.CONFIG)
        self.assertIn(self.DATASET, known_dataset_ids(), "the created file must be a valid metadata file")

    def test_existing_metadata_keeps_its_other_fields(self):
        service = self.service()
        service.metadata_manager.save_metadata({"doc_count": 3633, "query_count": 323}, self.DATASET)
        service.save_preprocessing_config(self.DATASET, self.CONFIG)
        metadata = service.get_metadata(self.DATASET)
        self.assertEqual((metadata["doc_count"], metadata["query_count"]), (3633, 323))
        self.assertEqual(metadata["preprocessing_config"], self.CONFIG)

    def test_saving_again_replaces_the_previous_config(self):
        service = self.service()
        service.save_preprocessing_config(self.DATASET, self.CONFIG)
        service.save_preprocessing_config(self.DATASET, {"use_lemmatization": False})
        self.assertEqual(service.get_preprocessing_config(self.DATASET), {"use_lemmatization": False})

    def test_an_empty_dataset_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service().save_preprocessing_config("", self.CONFIG)

    def test_the_query_side_reads_what_the_build_saved(self):
        """api_gateway preprocesses every query with this config: it must find it in the same metadata."""
        from services.api_gateway.core.dataset import preprocessing_config

        self.service().save_preprocessing_config(self.DATASET, self.CONFIG)
        self.assertEqual(preprocessing_config("data/metadata", self.DATASET), self.CONFIG)
        self.assertIsNone(preprocessing_config("data/metadata", None))
        self.assertIsNone(preprocessing_config("data/metadata", "nothing/here"))


if __name__ == "__main__":
    unittest.main()
