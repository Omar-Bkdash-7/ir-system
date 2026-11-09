"""The ONE place that turns a dataset id into a directory slug, a table name or a safe path.

Rules that must never change (they name things that already exist on disk / in PostgreSQL):
  index_slug  '/', '-' and '.' -> '_'                 (index directories: data/indices/<slug>/...; this is also what
                                                        common.config.get_dataset_specific_path and the retrieval
                                                        service's dataset_slug produce, so a '.' had to be replaced
                                                        here too: a dotted id was built in a directory retrieval
                                                        never looked in, and the index was invisible)
  table_slug  '/', '-', '.' -> '_', lower-cased       (existing tables; common.database.get_documents_by_ids builds
                                                        the same name, and PostgreSQL folds unquoted names to lower case)
"""
import re
from pathlib import Path
from typing import Iterable, List, Optional

from services.indexing.config import indices_base_path
from services.indexing.errors import InvalidDatasetIdError, InvalidIndexTypeError

INDEX_TYPES = ("tfidf", "bm25", "embedding")
PG_IDENTIFIER_LIMIT = 63  # PostgreSQL silently truncates longer identifiers
_SAFE_TABLE_SLUG = re.compile(r"^[a-z0-9_]+$")


def validate_index_type(index_type: str) -> str:
    if index_type not in INDEX_TYPES:
        raise InvalidIndexTypeError(f"Invalid index type: {index_type}")
    return index_type


def index_slug(dataset_name: str) -> str:
    """Directory name of a dataset below data/indices. Rejects '', '.', '..' (they would point outside)."""
    slug = dataset_name.replace("/", "_").replace("-", "_").replace(".", "_")
    # '', '.', '..' and '/' all collapse to nothing but underscores: a name with no letter or digit left cannot name
    # a directory.
    if not any(char.isalnum() for char in slug):
        raise InvalidDatasetIdError(f"Invalid dataset name: {dataset_name!r}")
    return slug


def index_dir(dataset_name: Optional[str], kind: str) -> Path:
    """data/indices/<slug>/<kind>; without a dataset name the old single-dataset layout (data/indices) is used."""
    base = indices_base_path()
    if dataset_name:
        return base / index_slug(dataset_name) / kind
    return base


def table_slug(dataset_id: str) -> str:
    return dataset_id.replace("/", "_").replace("-", "_").replace(".", "_").lower()


def table_name(dataset_id: str, prefix: str) -> str:
    """`<prefix>_<slug>` - exactly what common.database.get_documents_by_ids builds, so the rule must not change.

    The name is pasted into SQL, so anything but [a-z0-9_] is refused here instead of reaching the database."""
    slug = table_slug(dataset_id)
    if not _SAFE_TABLE_SLUG.match(slug):
        raise InvalidDatasetIdError(
            f"Dataset id {dataset_id!r} cannot be used as a database table name "
            f"(allowed characters: letters, digits, '/', '-', '.', '_')"
        )
    return f"{prefix}_{slug}"


def dataset_id_parts(dataset_id: str) -> List[str]:
    """'a/b' -> ['a', 'b']. Rejects what could escape a directory: empty ids, '.', '..', NUL."""
    if not dataset_id or not dataset_id.strip():
        raise InvalidDatasetIdError("Dataset ID cannot be empty")
    if "\x00" in dataset_id:
        raise InvalidDatasetIdError("Dataset ID contains an invalid character")
    parts = [p for p in dataset_id.strip("/").split("/") if p != ""]
    if not parts:
        raise InvalidDatasetIdError("Dataset ID cannot be empty")
    if any(p in (".", "..") for p in parts):
        raise InvalidDatasetIdError(f"Unsafe dataset id: {dataset_id!r}")
    return parts


def safe_child_path(root: Path, dataset_id: str) -> Path:
    """root/<id parts>, guaranteed to be STRICTLY inside `root` (also after resolving symlinks)."""
    target = root.joinpath(*dataset_id_parts(dataset_id))
    real_root, real_target = root.resolve(), target.resolve()
    if real_target == real_root or real_root not in real_target.parents:
        raise InvalidDatasetIdError(f"Dataset path escapes the datasets directory: {dataset_id!r}")
    return target


def table_to_dataset_id(table: str, prefix: str, known_ids: Iterable[str] = ()) -> str:
    """Best-effort inverse of table_name(). The mapping is lossy ('-', '_' and '.' all became '_'), so the exact ids
    from the metadata files win; the old heuristic (every '_' is a '/') is only the fallback."""
    for known in known_ids:
        try:
            candidate = table_name(known, prefix)
        except InvalidDatasetIdError:
            continue
        if candidate[:PG_IDENTIFIER_LIMIT] == table:  # PostgreSQL stores the truncated name
            return known
    stripped = table[len(prefix) + 1:] if table.startswith(f"{prefix}_") else table
    return stripped.replace("_", "/")
