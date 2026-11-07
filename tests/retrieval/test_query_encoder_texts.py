import os
import threading
import time
import unittest
from unittest import mock

import numpy as np

from common.exceptions import InvalidQueryError, ModelLoadError, RetrievalError
from services.retrieval.core.encoder import QueryEncoder
from services.retrieval.core.query import QueryContext
from services.retrieval.core.texts import TextStore, _connection_arguments, table_name
from tests.retrieval.helpers import FakeEncoder, tokens_of


class CountingPreprocess:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return tokens_of(text)


class QueryPolicyTests(unittest.TestCase):
    def make(self, text="", tokens=(), mode="preprocessed", preprocess=None):
        self.pre = preprocess or CountingPreprocess()
        self.encoder = FakeEncoder()
        return QueryContext(text=text, tokens=list(tokens), encoder=self.encoder, embedding_query_mode=mode, preprocess=self.pre)

    def test_tokens_that_were_sent_are_used_as_they_are(self):
        q = self.make(text="Boundary Layers!", tokens=["boundary", "layer"])
        self.assertEqual(q.lexical_tokens("bm25"), ("boundary", "layer"))
        self.assertEqual(q.lexical_tokens("tfidf"), ("boundary", "layer"))
        self.assertEqual(self.pre.calls, [])  # never preprocessed a second time

    def test_text_only_queries_are_preprocessed_once_for_all_models(self):
        q = self.make(text="Boundary Layers")
        q.lexical_tokens("bm25")
        q.lexical_tokens("tfidf")
        self.assertEqual(self.pre.calls, ["Boundary Layers"])

    def test_concurrent_callers_trigger_one_preprocessing(self):
        slow = CountingPreprocess()
        original = slow.__call__

        def delayed(text):
            time.sleep(0.05)
            return original(text)

        q = self.make(text="boundary layer", preprocess=delayed)
        threads = [threading.Thread(target=q.lexical_tokens, args=("bm25",)) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(slow.calls), 1)

    def test_blank_tokens_are_dropped(self):
        self.assertEqual(self.make(tokens=["a", "", "  ", "b"]).lexical_tokens("bm25"), ("a", "b"))

    def test_empty_query_is_rejected_for_every_model(self):
        q = self.make()
        for kind in ("bm25", "tfidf"):
            with self.assertRaisesRegex(InvalidQueryError, "query_tokens or query_text"):
                q.require(kind)
            with self.assertRaises(InvalidQueryError):
                q.lexical_tokens(kind)
        with self.assertRaisesRegex(InvalidQueryError, "query_text is required for embedding"):
            q.require("embedding")

    def test_embedding_needs_the_text_even_when_tokens_are_given(self):
        with self.assertRaises(InvalidQueryError):
            self.make(tokens=["a"]).require("embedding")

    def test_preprocessed_mode_embeds_the_given_tokens(self):
        q = self.make(text="Boundary Layers", tokens=["boundary", "layer"])
        q.dense_vector()
        self.assertEqual(self.encoder.calls, ["boundary layer"])
        self.assertEqual(self.pre.calls, [])

    def test_preprocessed_mode_without_tokens_preprocesses_the_text(self):
        self.make(text="Boundary Layers").dense_vector()
        self.assertEqual(self.encoder.calls, ["boundary layers"])

    def test_preprocessed_mode_falls_back_to_the_text_when_nothing_is_left(self):
        q = self.make(text="the of", preprocess=lambda t: [])
        q.dense_vector()
        self.assertEqual(self.encoder.calls, ["the of"])

    def test_raw_mode_embeds_the_original_text(self):
        q = self.make(text="  What is a Boundary Layer?  ", tokens=["boundary", "layer"], mode="raw")
        q.dense_vector()
        self.assertEqual(self.encoder.calls, ["What is a Boundary Layer?"])
        self.assertEqual(self.pre.calls, [])


class FakeModel:
    def __init__(self, dim=4, delay=0.0):
        self.dim, self.delay, self.calls, self.active, self.max_active = dim, delay, [], 0, 0
        self._lock = threading.Lock()

    def encode(self, sentences, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        self.calls.append((list(sentences), kwargs))
        return np.tile(np.arange(1, self.dim + 1, dtype=np.float64), (len(sentences), 1))


class EncoderTests(unittest.TestCase):
    def test_repeated_texts_are_served_from_the_cache(self):
        model = FakeModel()
        encoder = QueryEncoder(cache_size=2, model_factory=lambda: model, device="cpu", normalize=True)
        first = encoder.encode("a")
        self.assertIs(encoder.encode("a"), first)
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(first.flags.writeable)
        self.assertEqual(first.dtype, np.float32)

    def test_cache_is_bounded_and_least_recently_used_goes_first(self):
        model = FakeModel()
        encoder = QueryEncoder(cache_size=2, model_factory=lambda: model, device="cpu")
        for text in ("a", "b", "a", "c"):  # "b" is the least recently used when "c" arrives
            encoder.encode(text)
        before = len(model.calls)
        encoder.encode("a")
        self.assertEqual(len(model.calls), before)
        encoder.encode("b")
        self.assertEqual(len(model.calls), before + 1)

    def test_model_arguments(self):
        model = FakeModel()
        QueryEncoder(model_factory=lambda: model, device="cpu", normalize=False).encode("x")
        _, kwargs = model.calls[0]
        self.assertEqual(kwargs, {"normalize_embeddings": False, "convert_to_numpy": True, "device": "cpu", "show_progress_bar": False})

    def test_model_loads_once_and_encodes_one_at_a_time(self):
        model, built = FakeModel(delay=0.02), []

        def factory():
            built.append(1)
            time.sleep(0.05)
            return model

        encoder = QueryEncoder(cache_size=0, model_factory=factory, device="cpu")
        threads = [threading.Thread(target=encoder.encode, args=(f"text {i}",)) for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(built), 1)
        self.assertEqual(model.max_active, 1)

    def test_errors(self):
        class Broken:
            def encode(self, *a, **k):
                raise RuntimeError("cuda out of memory")

        with self.assertRaises(RetrievalError):
            QueryEncoder(model_factory=Broken, device="cpu").encode("x")

        def unavailable():
            raise ModelLoadError("no model")

        with self.assertRaises(ModelLoadError):
            QueryEncoder(model_factory=unavailable, device="cpu").encode("x")

    def test_info_never_loads_the_model(self):
        encoder = QueryEncoder(model_factory=lambda: self.fail("model must not load"), device="cpu")
        info = encoder.info()
        self.assertEqual(info["device"], "cpu")
        self.assertIn("model_name", info)
        self.assertFalse(encoder.model_loaded)


class TextStoreTests(unittest.TestCase):
    def test_table_names_follow_the_indexing_rule(self):
        self.assertEqual(table_name("beir/quora", "dataset"), "dataset_beir_quora")
        self.assertEqual(table_name("Beir/TREC-Covid.v2", "dataset"), "dataset_beir_trec_covid_v2")
        self.assertEqual(table_name("cranfield", "docs"), "docs_cranfield")

    def test_unsafe_dataset_names_never_reach_sql(self):
        for bad in ("x; DROP TABLE y", "a b", "a'b", "ü"):
            with self.assertRaises(InvalidQueryError):
                table_name(bad, "dataset")

    def test_fetch_uses_the_injected_fetcher_and_skips_empty_requests(self):
        calls = []
        store = TextStore(fetcher=lambda dataset, ids: calls.append((dataset, list(ids))) or {"a": "text a"})
        self.assertEqual(store.fetch("toy", []), {})
        self.assertEqual(calls, [])
        self.assertEqual(store.fetch("toy", ["a", "b"]), {"a": "text a"})
        self.assertEqual(calls, [("toy", ["a", "b"])])

    def test_connection_arguments_prefer_database_url(self):
        with mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://u:p@host:5432/db"}):
            self.assertEqual(_connection_arguments(), {"dsn": "postgresql://u:p@host:5432/db"})
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("DATABASE_URL", None)
            arguments = _connection_arguments()
        self.assertEqual(set(arguments), {"host", "port", "user", "password", "database"})


if __name__ == "__main__":
    unittest.main()
