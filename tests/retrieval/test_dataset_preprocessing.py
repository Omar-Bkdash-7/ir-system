"""A text-only query (what the RAG service sends) must be preprocessed with the dataset's saved config.

Regression: the retrieval service preprocessed every text-only query with the DEFAULTS, while the documents of that
dataset may have been indexed with a custom config. api_gateway already read the saved config, so the UI and the RAG
service searched the same index with differently tokenised queries.
"""
import contextlib
import json
import sys
import types
import unittest

from services.retrieval.config import RetrievalSettings
from services.retrieval.core.compact import CompactCache
from services.retrieval.core.query import default_preprocess, preprocessor_for, saved_preprocessing_config
from services.retrieval.core.service import RetrievalService
from services.retrieval.core.store import IndexStore
from tests.retrieval.helpers import TOY_DOCS, FakeEncoder, FakeTexts, project_dir, request, write_bm25

MODULE = "services.preprocessing.preprocessing_service"
CONFIG = {"use_stemming": True, "remove_stopwords": True}
_MISSING = object()


@contextlib.contextmanager
def fake_preprocessing(calls):
    """Replace the preprocessing service for the block.

    mock.patch.dict(sys.modules, ...) restores the whole dict, deleting everything imported inside the block, so only
    the keys added here are removed again.
    """
    module = types.ModuleType(MODULE)

    class PreprocessingConfig:
        def __init__(self, **values):
            self.values = values

        @classmethod
        def from_dict(cls, data):
            return cls(**data)

    def preprocess(text, mode=None, config=None):
        calls.append(config)
        return ["boundary"]

    module.PreprocessingConfig = PreprocessingConfig
    module.preprocess = preprocess
    previous = sys.modules.get(MODULE, _MISSING)
    sys.modules[MODULE] = module
    try:
        yield PreprocessingConfig
    finally:
        if previous is _MISSING:
            sys.modules.pop(MODULE, None)
        else:
            sys.modules[MODULE] = previous


class DatasetPreprocessingTest(unittest.TestCase):
    def setUp(self):
        self._ctx = project_dir()
        self.root = self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__, None, None, None)
        metadata = self.root / "data" / "metadata" / "toy.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({"dataset_name": "toy", "preprocessing_config": CONFIG}), encoding="utf-8")

    def service(self):
        store = IndexStore(cache=CompactCache(None), stale_check_seconds=0.0)
        service = RetrievalService(RetrievalSettings(stale_check_seconds=0.0), store, FakeEncoder(), FakeTexts({}))
        self.addCleanup(service.close)
        return service

    def test_the_saved_config_is_read_and_missing_metadata_is_not_an_error(self):
        self.assertEqual(saved_preprocessing_config("toy"), CONFIG)
        self.assertIsNone(saved_preprocessing_config("other"))
        self.assertIsNone(saved_preprocessing_config(None))

    def test_a_dataset_without_a_saved_config_uses_the_defaults(self):
        self.assertIs(preprocessor_for("other"), default_preprocess)
        self.assertIs(preprocessor_for(None), default_preprocess)

    def test_a_text_only_query_is_preprocessed_with_the_dataset_config(self):
        write_bm25(self.root, "toy", TOY_DOCS)
        service = self.service()
        calls = []
        with fake_preprocessing(calls) as PreprocessingConfig:
            outcome = service.retrieve(request(query_tokens=[], query_text="wing flutter"))
        self.assertTrue(outcome.results)
        self.assertEqual(len(calls), 1)  # the query is preprocessed once, not once per model
        self.assertIsInstance(calls[0], PreprocessingConfig)
        self.assertEqual(calls[0].values, CONFIG)


if __name__ == "__main__":
    unittest.main()
