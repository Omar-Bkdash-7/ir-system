"""Path rules of the clustering service (one place, so the cache and the index always agree).

The dataset -> directory slug is the indexing service's rule, imported rather than re-implemented: a clustering run
for 'beir/quora' must read data/indices/beir_quora/embedding, exactly the directory indexing wrote, and it must not
create data/clusters/beir/quora (which used to hide the run from list_clustered_datasets and break /delete).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from services.clustering.errors import InvalidClusteringParameterError
from services.indexing.naming import index_slug
from services.indexing.errors import InvalidDatasetIdError


def dataset_slug(dataset: str) -> str:
    """Directory name of the dataset: '/', '-' and '.' become '_' (services.indexing.naming.index_slug)."""
    try:
        return index_slug(dataset.strip())
    except InvalidDatasetIdError as exc:
        raise InvalidClusteringParameterError(str(exc)) from exc


def dataset_dir(data_dir: Path, dataset: str) -> Path:
    """<data_dir>/<slug> - one directory per dataset, never nested."""
    return Path(data_dir) / dataset_slug(dataset)


def embedding_files(indices_dir: Path, dataset: str) -> Tuple[Path, Path]:
    """(embeddings.npy, doc_id_map.json) below <indices_dir>/<slug>/embedding."""
    base = Path(indices_dir) / dataset_slug(dataset) / "embedding"
    return base / "embeddings.npy", base / "doc_id_map.json"


def run_artifact(directory: Path, run_id: str, kind: str) -> Path:
    """Derived file of one run: <directory>/run-<run_id>-<kind>.

    Derived data (2D projection, topics) is named after the run that produced it, so a new run can never be read
    together with half of an older one: the manifest names one run_id and readers only open its files.
    """
    return Path(directory) / f"run-{run_id}-{kind}"
