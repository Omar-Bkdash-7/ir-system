"""A broken embedding index must fail early with a clear error, and its fingerprint must follow the file."""
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from services.clustering.core.embeddings import index_fingerprint, load_embedding_index
from services.clustering.errors import EmbeddingIndexMissingError, InvalidClusteringParameterError
from tests.clustering.helpers import write_embedding_index


class EmbeddingIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.indices = Path(self.tmp.name) / "indices"
        self.embeddings = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
        self.doc_ids = [f"d{i}" for i in range(6)]

    def _write(self, embeddings=None, doc_ids=None, dataset="cranfield"):
        return write_embedding_index(
            self.indices,
            dataset,
            self.embeddings if embeddings is None else embeddings,
            self.doc_ids if doc_ids is None else doc_ids,
        )

    def test_a_valid_index_is_loaded_and_cast_to_float32(self):
        self._write(embeddings=self.embeddings.astype(np.float64))
        index = load_embedding_index(self.indices, "cranfield")
        self.assertEqual(index.n_docs, 6)
        self.assertEqual(index.dimension, 4)
        self.assertEqual(index.doc_ids, self.doc_ids)
        self.assertEqual(index.embeddings.dtype, np.float32)

    def test_a_missing_index_is_a_404_style_error(self):
        with self.assertRaises(EmbeddingIndexMissingError):
            load_embedding_index(self.indices, "cranfield")

    def test_a_missing_doc_id_map_is_reported_by_name(self):
        base = self._write()
        (base / "doc_id_map.json").unlink()
        with self.assertRaises(EmbeddingIndexMissingError) as raised:
            load_embedding_index(self.indices, "cranfield")
        self.assertIn("doc_id_map.json", str(raised.exception))

    def test_a_count_mismatch_is_refused(self):
        self._write(doc_ids=self.doc_ids[:3])
        with self.assertRaises(InvalidClusteringParameterError):
            load_embedding_index(self.indices, "cranfield")

    def test_nan_values_are_refused(self):
        broken = self.embeddings.copy()
        broken[0, 0] = np.nan
        self._write(embeddings=broken)
        with self.assertRaises(InvalidClusteringParameterError):
            load_embedding_index(self.indices, "cranfield")

    def test_a_one_dimensional_matrix_is_refused(self):
        self._write(embeddings=np.arange(6, dtype=np.float32), doc_ids=self.doc_ids)
        with self.assertRaises(InvalidClusteringParameterError):
            load_embedding_index(self.indices, "cranfield")

    def test_a_non_list_doc_id_map_is_refused(self):
        base = self._write()
        (base / "doc_id_map.json").write_text(json.dumps({"doc0": 0}), encoding="utf-8")
        with self.assertRaises(InvalidClusteringParameterError):
            load_embedding_index(self.indices, "cranfield")

    def test_the_fingerprint_does_not_need_the_matrix(self):
        self._write()
        self.assertEqual(
            index_fingerprint(self.indices, "cranfield"),
            load_embedding_index(self.indices, "cranfield").fingerprint,
        )

    def test_the_fingerprint_changes_when_the_index_is_rebuilt(self):
        base = self._write()
        before = index_fingerprint(self.indices, "cranfield")
        time.sleep(0.01)
        np.save(base / "embeddings.npy", self.embeddings * 2.0)
        time.sleep(0.01)
        self.assertNotEqual(before, index_fingerprint(self.indices, "cranfield"))

    def test_a_dotted_dataset_uses_the_indexed_directory(self):
        self._write(dataset="trec.covid")
        self.assertEqual(load_embedding_index(self.indices, "trec.covid").n_docs, 6)
