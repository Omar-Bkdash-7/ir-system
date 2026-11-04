"""The package's public surface: every old import path keeps working, importing the package stays cheap.

Each check runs in a fresh interpreter, because what matters is what an import pulls in."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

EXPECTED_ALL = {
    "services.indexing": ["DatasetService", "build_bm25_index", "build_embedding_index", "build_tfidf_index",
                          "delete_local_dataset", "get_bm25_stats", "get_db_dataset_info", "get_embedding_stats",
                          "get_index_stats", "get_metadata", "get_tfidf_stats", "list_local_datasets",
                          "upload_dataset_to_db", "validate_and_store_metadata"],
    "services.indexing.dataset": ["DATA_FILE_EXTENSIONS", "DEFAULT_BATCH_SIZE", "DEFAULT_RETRY_DELAY",
                                  "DEFAULT_VALIDATION_RETRIES", "DatabaseInfoRetriever", "DatasetDatabaseDeleter",
                                  "DatasetLoader", "DatasetMetadataManager", "DatasetPathResolver", "DatasetService",
                                  "DatasetUploader", "DatasetValidator", "LocalDatasetDeleter", "LocalDatasetScanner",
                                  "database_cursor", "delete_dataset_from_db", "delete_local_dataset",
                                  "get_db_dataset_info", "get_metadata", "list_local_datasets", "upload_dataset_to_db",
                                  "validate_and_store_metadata"],
    "services.indexing.services": ["BM25IndexBuilder", "BM25StatsRetriever", "EmbeddingIndexBuilder",
                                   "EmbeddingStatsRetriever", "TFIDFIndexBuilder", "TFIDFStatsRetriever",
                                   "build_all_indices", "build_bm25_index", "build_embedding_index",
                                   "build_tfidf_index", "get_bm25_stats", "get_embedding_stats", "get_index_stats",
                                   "get_tfidf_stats"],
    "services.indexing.indexers": ["BM25Indexer", "EmbeddingIndexer", "TFIDFIndexer"],
}

# every import path used by something in the repository (common/dataset_context.py, scripts/, the old shim modules)
IMPORT_PATHS = [
    "from services.indexing import DatasetService, get_index_stats, build_bm25_index, validate_and_store_metadata",
    "from services.indexing.dataset import DatasetValidator, DatasetLoader, DatasetMetadataManager, LocalDatasetScanner",
    "from services.indexing.dataset import LocalDatasetDeleter, DatasetUploader, DatabaseInfoRetriever, "
    "DatasetDatabaseDeleter, database_cursor",
    "from services.indexing.dataset import DatasetPathResolver, DEFAULT_BATCH_SIZE, DATA_FILE_EXTENSIONS",
    "from services.indexing.dataset.constants import DATA_FILE_EXTENSIONS, MAX_RECURSION_DEPTH",
    "from services.indexing.dataset.scanner import LocalDatasetScanner",
    "from services.indexing.dataset.loader import DatasetLoader",
    "from services.indexing.dataset.path_resolver import DatasetPathResolver",
    "from services.indexing.dataset.metadata_manager import get_metadata, list_all_datasets, DatasetMetadataManager",
    "from services.indexing.indexers import TFIDFIndexer, BM25Indexer, EmbeddingIndexer",
    "from services.indexing.services import build_all_indices, get_index_stats, get_tfidf_stats",
    "from services.indexing.services.build_service import TFIDFIndexBuilder, EmbeddingIndexBuilder, BM25IndexBuilder",
    "from services.indexing.services.stats_service import TFIDFStatsRetriever, BM25StatsRetriever",
    "from services.indexing.services.inverted_index_service import get_inverted_index_stats, get_term_info, "
    "search_terms, get_vocabulary_sample",
    "from services.indexing.dataset_service import validate_and_store_metadata, DatasetService, delete_dataset_from_db",
    "from services.indexing.indexing_service import get_tfidf_status, get_embedding_status, get_bm25_status, "
    "build_all_indices",
]


def run_python(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(os.path.abspath(p or ".") for p in sys.path))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          cwd=tempfile.gettempdir())


class PackageContractTest(unittest.TestCase):
    def test_every_exported_name_exists_and_resolves(self):
        code = (
            "import importlib, json\n"
            f"expected = {json.dumps(EXPECTED_ALL)}\n"
            "problems = []\n"
            "for module, names in expected.items():\n"
            "    mod = importlib.import_module(module)\n"
            "    if sorted(mod.__all__) != names: problems.append(f'{module}.__all__ changed')\n"
            "    for name in names:\n"
            "        try: getattr(mod, name)\n"
            "        except Exception as e: problems.append(f'{module}.{name}: {type(e).__name__}: {e}')\n"
            "print(json.dumps(problems))\n"
        )
        result = run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), [])

    def test_every_import_path_used_in_the_repository_still_works(self):
        code = (
            "import json, warnings\n"
            "warnings.simplefilter('ignore')\n"
            f"paths = {json.dumps(IMPORT_PATHS)}\n"
            "failed = {}\n"
            "for stmt in paths:\n"
            "    try: exec(stmt, {})\n"
            "    except Exception as e: failed[stmt] = f'{type(e).__name__}: {e}'\n"
            "print(json.dumps(failed))\n"
        )
        result = run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), {})

    def test_what_common_dataset_context_imports_is_cheap(self):
        # common/dataset_context.py runs `from services.indexing.dataset.metadata_manager import get_metadata` inside
        # every service that asks for dataset metadata. It used to import scikit-learn, scipy, numpy, psycopg2 and
        # ir_datasets (seconds, hundreds of MB) and to raise a DeprecationWarning.
        code = (
            "import json, sys, warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    from services.indexing.dataset.metadata_manager import get_metadata, list_all_datasets\n"
            "heavy = [m for m in ('sklearn', 'scipy', 'numpy', 'psycopg2', 'ir_datasets', 'jsonlines', 'torch') if m in sys.modules]\n"
            "ours = [str(w.message) for w in caught if 'indexing' in str(w.filename).replace(chr(92), '/')]\n"
            "print(json.dumps({'heavy': heavy, 'warnings': ours}))\n"
        )
        result = run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), {"heavy": [], "warnings": []})


if __name__ == "__main__":
    unittest.main()
