"""The three indexers: on-disk layout (the retrieval service reads these files), crash-safe saving, consistency."""
import json
import pickle
from pathlib import Path
from unittest import mock

import numpy as np

from services.indexing.indexers import BM25Indexer, EmbeddingIndexer, TFIDFIndexer
from tests.indexing._support import DOCS, WorkdirTestCase


def _partial_pickle_dump(obj, fp, *args, **kwargs):
    fp.write(b"garbage")
    raise OSError("disk full")


class BM25IndexerTest(WorkdirTestCase):
    def test_files_are_where_they_have_always_been(self):
        self.assertEqual(BM25Indexer("beir/quora").index_path, Path("data/indices/beir_quora/bm25/bm25_index.pkl"))
        self.assertEqual(BM25Indexer("nano-beir/quora").index_path,
                         Path("data/indices/nano_beir_quora/bm25/bm25_index.pkl"))
        self.assertEqual(BM25Indexer().index_path, Path("data/indices/bm25_index.pkl"))  # old single-dataset layout

    def test_build_save_load_roundtrip(self):
        BM25Indexer("demo").build(self.processed()).save()
        index = BM25Indexer("demo").load()
        self.assertTrue(BM25Indexer("demo").exists())
        self.assertEqual(index.total_docs, 4)
        self.assertEqual(index.doc_ids, ["d1", "d2", "d3", "d4"])
        self.assertEqual(index.doc_frequencies["flow"], 2)
        self.assertAlmostEqual(index.avg_doc_length, 3.25)
        self.assertGreater(index.idf_scores["heat"], index.idf_scores["flow"])

    def test_pickle_format_is_unchanged(self):
        # other services read this file directly, so the keys and the posting layout are a contract
        BM25Indexer("demo").build(self.processed()).save()
        with open(BM25Indexer("demo").index_path, "rb") as fp:
            data = pickle.load(fp)
        self.assertEqual(set(data), {"inverted_index", "doc_frequencies", "idf_scores", "doc_lengths",
                                     "avg_doc_length", "total_docs", "doc_ids"})
        self.assertEqual(data["inverted_index"]["flow"], [("d2", 1), ("d3", 2)])

    def test_a_second_build_on_the_same_instance_starts_from_scratch(self):
        index = BM25Indexer("demo")
        index.build(self.processed("a.jsonl"))
        index.build(self.processed("b.jsonl", DOCS[:2]))
        self.assertEqual(index.total_docs, 2)
        self.assertEqual(index.doc_ids, ["d1", "d2"])
        self.assertNotIn("heating", index.inverted_index)  # a term that only existed in the first corpus

    def test_a_crash_while_saving_keeps_the_previous_index(self):
        BM25Indexer("demo").build(self.processed()).save()
        other = BM25Indexer("demo").build(self.processed("b.jsonl", DOCS[:1]))
        with mock.patch("pickle.dump", side_effect=_partial_pickle_dump), self.assertRaises(OSError):
            other.save()
        self.assertEqual(BM25Indexer("demo").load().total_docs, 4)  # the old, complete index is still there
        self.assertEqual([p.name for p in BM25Indexer("demo").index_path.parent.iterdir()], ["bm25_index.pkl"])

    def test_unsafe_dataset_names_cannot_leave_the_indices_directory(self):
        for bad in ("..", "."):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                BM25Indexer(bad)


class TFIDFIndexerTest(WorkdirTestCase):
    def test_files_are_where_they_have_always_been(self):
        indexer = TFIDFIndexer("nano-beir/quora")
        base = Path("data/indices/nano_beir_quora/tfidf")
        self.assertEqual(indexer.vectorizer_path, base / "tfidf_vectorizer.pkl")
        self.assertEqual(indexer.matrix_path, base / "tfidf_matrix.npz")
        self.assertEqual(indexer.doc_ids_path, base / "tfidf_doc_ids.json")
        self.assertEqual(TFIDFIndexer().matrix_path, Path("data/indices/tfidf_matrix.npz"))

    def test_build_save_load_roundtrip(self):
        TFIDFIndexer("demo").build(self.processed()).save()
        index = TFIDFIndexer("demo").load()
        self.assertEqual(index.matrix.shape[0], 4)
        self.assertEqual(index.doc_ids, ["d1", "d2", "d3", "d4"])
        # Every preprocessed token is indexed exactly as it is (BM25 takes the same tokens). scikit-learn's default
        # pattern used to re-tokenise the corpus: one-character tokens ('x', 'a') were dropped and compound tokens
        # were split, so the two models indexed different vocabularies.
        expected = sorted({token for document in DOCS for token in document["tokens"]})
        self.assertEqual(sorted(index.vectorizer.vocabulary_), expected)

    def test_inconsistent_files_are_refused_instead_of_answering_wrongly(self):
        TFIDFIndexer("demo").build(self.processed()).save()
        TFIDFIndexer("demo").doc_ids_path.write_text(json.dumps(["d1"]))  # e.g. left over by an interrupted build
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            TFIDFIndexer("demo").load()

    def test_a_crash_while_saving_keeps_the_previous_files(self):
        TFIDFIndexer("demo").build(self.processed()).save()
        again = TFIDFIndexer("demo").build(self.processed())
        with mock.patch("pickle.dump", side_effect=_partial_pickle_dump), self.assertRaises(OSError):
            again.save()
        self.assertEqual(TFIDFIndexer("demo").load().matrix.shape[0], 4)
        leftovers = [p.name for p in TFIDFIndexer("demo").matrix_path.parent.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class EmbeddingIndexerTest(WorkdirTestCase):
    def _with_fake_model(self, dataset="demo"):
        indexer = EmbeddingIndexer(dataset)
        indexer.model = mock.Mock()
        indexer.model.get_sentence_embedding_dimension.return_value = 3
        indexer.model.encode.side_effect = (
            lambda texts, **kwargs: np.arange(1, len(texts) * 3 + 1, dtype="float32").reshape(len(texts), 3))
        return indexer

    def test_files_are_where_they_have_always_been(self):
        indexer = EmbeddingIndexer("nano-beir/quora")
        self.assertEqual(indexer.embeddings_path, Path("data/indices/nano_beir_quora/embedding/embeddings.npy"))
        self.assertEqual(indexer.doc_id_map_path, Path("data/indices/nano_beir_quora/embedding/doc_id_map.json"))

    def test_build_save_load_roundtrip(self):
        self._with_fake_model().build(self.processed()).save()
        index = EmbeddingIndexer("demo").load()
        self.assertEqual(index.embeddings.shape, (4, 3))
        self.assertEqual(index.doc_ids, ["d1", "d2", "d3", "d4"])

    def test_status_pages_can_map_the_matrix_instead_of_reading_it(self):
        self._with_fake_model().build(self.processed()).save()
        index = EmbeddingIndexer("demo").load(mmap=True)
        self.assertIsInstance(index.embeddings, np.memmap)
        self.assertEqual(index.embeddings.shape, (4, 3))

    def test_the_gpu_is_only_probed_when_the_device_is_needed(self):
        with mock.patch("services.indexing.indexers.embedding_indexer.get_torch_device", return_value="cpu") as probe:
            indexer = EmbeddingIndexer("demo")
            probe.assert_not_called()  # status endpoints construct indexers just to look at files
            self.assertEqual(indexer.device, "cpu")
            self.assertEqual(indexer.device, "cpu")
            probe.assert_called_once()

    def test_a_crash_while_saving_keeps_the_previous_index(self):
        self._with_fake_model().build(self.processed()).save()
        again = self._with_fake_model().build(self.processed())

        def partial_save(fp, array, *args, **kwargs):
            fp.write(b"garbage")
            raise OSError("disk full")

        with mock.patch("numpy.save", side_effect=partial_save), self.assertRaises(OSError):
            again.save()
        self.assertEqual(EmbeddingIndexer("demo").load().embeddings.shape, (4, 3))


if __name__ == "__main__":
    import unittest
    unittest.main()
