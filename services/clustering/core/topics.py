"""Topic keywords and representative documents for each cluster.

Keywords are computed with c-TF-IDF: one document per cluster (all its texts concatenated), then TF-IDF across the
clusters. That is the class-based TF-IDF that makes a word which is common *inside* one cluster and rare in the others
rank high, and it is computed here from the clusters we actually produced.

BERTopic was removed from this path on purpose: it re-clustered the documents with its own HDBSCAN (31 topics for a
2-cluster run) and its topic ids were therefore unrelated to our cluster ids - `get_topic(cluster_id)` returned the
keywords of some other cluster. c-TF-IDF gives the same kind of output, deterministically and without a model.

Representative documents are the ones closest to their cluster centroid: "the longest three texts" (the previous
rule) says nothing about whether a text represents its cluster.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from services.clustering.schemas import TopicInfo

logger = logging.getLogger(__name__)

#: Letters only, at least three characters: a keyword list made of "595" or "58" tells a reader nothing.
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"
STOP_WORDS = "english"


@dataclass(frozen=True)
class TopicExtraction:
    """Topics per cluster id (noise excluded) and which extractor produced them."""

    topics: Dict[int, TopicInfo]
    method: str  # "ctfidf" | "none"

    @property
    def available(self) -> bool:
        return self.method != "none"


def _cluster_indices(labels: np.ndarray) -> Dict[int, np.ndarray]:
    clusters: Dict[int, np.ndarray] = {}
    for cluster_id in np.unique(labels):
        cluster_id = int(cluster_id)
        if cluster_id == -1:
            continue
        clusters[cluster_id] = np.flatnonzero(labels == cluster_id)
    return clusters


def _centroid_representatives(
    embeddings: np.ndarray,
    texts: Sequence[str],
    indices: np.ndarray,
    count: int,
) -> List[str]:
    """The `count` documents closest to their cluster centroid (cosine), falling back to text length if the
    cluster's vectors carry no direction (all zeros)."""
    if count <= 0 or len(indices) == 0:
        return []

    vectors = embeddings[indices]
    centroid = vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm == 0.0:
        ordered = sorted(indices.tolist(), key=lambda i: len(texts[i]), reverse=True)
    else:
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0.0] = 1.0
        similarity = (vectors @ centroid) / (norms * centroid_norm)
        # Stable sort: ties keep the index order, so the same run always picks the same documents.
        ordered = [int(indices[i]) for i in np.argsort(-similarity, kind="stable")[:count]]

    return [texts[i] for i in ordered[:count]]


def ctfidf_keywords(texts_by_cluster: Dict[int, Sequence[str]], n_keywords: int) -> Dict[int, List[str]]:
    """Top c-TF-IDF words per cluster (see the module docstring). Missing/blank texts are ignored."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_ids = sorted(texts_by_cluster)
    documents = [
        " ".join(text for text in texts_by_cluster[cluster_id] if text and text.strip()) for cluster_id in cluster_ids
    ]
    if not any(document.strip() for document in documents):
        return {cluster_id: [] for cluster_id in cluster_ids}

    try:
        vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS,
            token_pattern=TOKEN_PATTERN,
            sublinear_tf=True,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:  # empty vocabulary after stop-word removal
        logger.warning("c-TF-IDF found no usable word in any cluster (%s)", exc)
        return {cluster_id: [] for cluster_id in cluster_ids}

    names = vectorizer.get_feature_names_out()
    keywords: Dict[int, List[str]] = {}
    for row, cluster_id in enumerate(cluster_ids):
        scores = np.asarray(matrix.getrow(row).todense()).ravel()
        order = np.argsort(-scores, kind="stable")[: max(1, int(n_keywords))]
        keywords[cluster_id] = [str(names[i]) for i in order if scores[i] > 0]
    return keywords


def extract_topics(
    texts: Sequence[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    n_keywords: int = 10,
    representative_count: int = 3,
) -> TopicExtraction:
    """Keywords + representative documents per cluster. Never raises: topics are secondary to the clustering."""
    if labels.shape[0] != len(texts) or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("texts, embeddings and labels must describe the same documents")

    clusters = _cluster_indices(labels)
    if not clusters or not any(text and text.strip() for text in texts):
        if clusters:
            logger.warning("No document text available; skipping topic extraction")
        return TopicExtraction(topics={}, method="none")

    keywords = ctfidf_keywords(
        {cluster_id: [texts[i] for i in indices] for cluster_id, indices in clusters.items()},
        n_keywords,
    )
    method = "ctfidf" if any(keywords.values()) else "none"

    topics: Dict[int, TopicInfo] = {}
    for cluster_id, indices in clusters.items():
        topics[cluster_id] = TopicInfo(
            topic_id=cluster_id,
            keywords=keywords.get(cluster_id, []),
            representative_docs=_centroid_representatives(embeddings, texts, indices, representative_count),
        )

    logger.info("Extracted topics for %d clusters using %s", len(topics), method)
    return TopicExtraction(topics=topics, method=method)
