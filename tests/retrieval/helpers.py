"""Shared test helpers: tiny indexes written in EXACTLY the formats the indexing service produces."""
import contextlib
import json
import math
import os
import pickle
import re
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np

TOY_DOCS = {
    "d1": "boundary layer flow over a flat plate at high speed",
    "d2": "heat transfer in a turbulent boundary layer",
    "d3": "supersonic wing flutter and aeroelastic stability",
    "d4": "shock wave interaction with a laminar boundary layer",
    "d5": "numerical solution of the navier stokes equations for viscous flow",
    "d6": "thin airfoil theory and lift of a flat plate",
    "d7": "wing flutter model tests in a wind tunnel",
    "d8": "heat conduction in a slab with variable properties",
}


def tokens_of(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


@contextlib.contextmanager
def project_dir():
    """Run in an empty temporary 'project root': the indices are looked up in ./data/indices/<dataset>/..."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(old)


def write_bm25(root: Path, dataset: str, docs: Dict[str, str], *, include_idf: bool = True) -> Path:
    """The pickle written by services.indexing.indexers.bm25_indexer."""
    directory = root / "data" / "indices" / dataset / "bm25"
    directory.mkdir(parents=True, exist_ok=True)
    doc_ids = list(docs)
    inverted: Dict[str, list] = {}
    lengths = {}
    for doc_id, text in docs.items():
        tokens = tokens_of(text)
        lengths[doc_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            inverted.setdefault(term, []).append((doc_id, tf))
    n = len(doc_ids)
    df = {term: len(entries) for term, entries in inverted.items()}
    idf = {term: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for term, d in df.items()}
    data = {
        "inverted_index": inverted,
        "doc_frequencies": df,
        "idf_scores": idf if include_idf else {},
        "doc_lengths": lengths,
        "avg_doc_length": sum(lengths.values()) / n,
        "total_docs": n,
        "doc_ids": doc_ids,
    }
    path = directory / "bm25_index.pkl"
    with open(path, "wb") as handle:
        pickle.dump(data, handle)
    return path


def write_tfidf(root: Path, dataset: str, docs: Dict[str, str], **vectorizer_args):
    """vectorizer pickle + L2-normalised CSR matrix + doc id list, as services.indexing.indexers.tfidf_indexer writes them."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer

    directory = root / "data" / "indices" / dataset / "tfidf"
    directory.mkdir(parents=True, exist_ok=True)
    args = {"sublinear_tf": True, "norm": "l2"}
    args.update(vectorizer_args)
    vectorizer = TfidfVectorizer(**args)
    matrix = vectorizer.fit_transform([" ".join(tokens_of(text)) for text in docs.values()])
    sp.save_npz(directory / "tfidf_matrix.npz", matrix)
    with open(directory / "tfidf_vectorizer.pkl", "wb") as handle:
        pickle.dump(vectorizer, handle)
    with open(directory / "tfidf_doc_ids.json", "w", encoding="utf-8") as handle:
        json.dump(list(docs), handle)
    return vectorizer, matrix


def hash_embedding(text: str, dim: int = 32) -> np.ndarray:
    """Deterministic bag-of-words 'embedding': texts that share words are close (a stand-in for the neural model)."""
    vector = np.zeros(dim, dtype=np.float64)
    for token in tokens_of(text):
        vector[sum(ord(c) * (i + 1) for i, c in enumerate(token)) % dim] += 1.0
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).astype(np.float32)


def write_embeddings(root: Path, dataset: str, docs: Dict[str, str], dim: int = 32) -> np.ndarray:
    directory = root / "data" / "indices" / dataset / "embedding"
    directory.mkdir(parents=True, exist_ok=True)
    matrix = np.vstack([hash_embedding(text, dim) for text in docs.values()]).astype(np.float32)
    np.save(directory / "embeddings.npy", matrix)
    with open(directory / "doc_id_map.json", "w", encoding="utf-8") as handle:
        json.dump(list(docs), handle)
    return matrix


class FakeEncoder:
    """Stands in for QueryEncoder: no neural model, counts its calls."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.calls: List[str] = []

    def encode(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return hash_embedding(text, self.dim)

    def info(self):
        return {"device": "cpu", "model_name": "fake", "local_path": None}

    def warm(self):
        pass


class FakeTexts:
    def __init__(self, texts: Dict[str, str], fail: bool = False):
        self.texts, self.fail, self.calls = texts, fail, []

    def fetch(self, dataset, doc_ids):
        self.calls.append(list(doc_ids))
        if self.fail:
            raise ConnectionError("database is down")
        return {doc_id: self.texts[doc_id] for doc_id in doc_ids if doc_id in self.texts}

    def close(self):
        pass


def request(**fields):
    """A request object with the attribute names of the pydantic request models."""
    base = dict(
        query_tokens=[], query_text="", model="bm25", top_k=5, include_text=False, bm25_k1=1.5, bm25_b=0.75,
        dataset="toy", first_model="bm25", second_model="tfidf", first_stage_k=100, final_top_k=5,
        models=["bm25", "tfidf"], fusion_method="rrf", model_weights=None, candidates_per_model=100, rrf_k=60,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def bm25_reference(docs: Dict[str, str], query_tokens: Sequence[str], k1=1.5, b=0.75) -> Dict[str, float]:
    """Plain-Python Okapi BM25, written independently of the code under test."""
    tokenised = {d: tokens_of(t) for d, t in docs.items()}
    n = len(docs)
    avg = sum(len(t) for t in tokenised.values()) / n
    scores: Dict[str, float] = {}
    for term in query_tokens:  # a repeated query word counts again
        df = sum(1 for t in tokenised.values() if term in t)
        if df == 0:
            continue
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        for doc_id, tokens in tokenised.items():
            tf = tokens.count(term)
            if tf:
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / avg))
    return scores
