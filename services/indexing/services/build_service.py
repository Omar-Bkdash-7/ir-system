"""
Index building service.

Provides functionality for building TF-IDF, Embedding, and BM25 indices
with force rebuild capability, preprocessing integration, and error handling.
"""

import functools
import logging
import jsonlines
from contextlib import closing
from pathlib import Path
from typing import Dict, Any, Optional
from tempfile import NamedTemporaryFile

from services.indexing.config import table_prefix as configured_table_prefix
from services.indexing.dataset.metadata_manager import known_dataset_ids
from services.indexing.indexers import TFIDFIndexer, EmbeddingIndexer, BM25Indexer
from services.indexing.naming import index_slug, table_name as build_table_name, table_slug, table_to_dataset_id
from services.indexing.storage import try_lock
from common.config import get_config


logger = logging.getLogger(__name__)

SERVER_CURSOR_ROWS = 2000  # rows fetched per round trip when a dataset is streamed out of PostgreSQL


def _already_running(label: str, dataset_name: str) -> Dict[str, str]:
    logger.warning(f"{label} build for '{dataset_name}' is already running, request ignored")
    return {
        "status": "skipped",
        "message": f"A {label} build for this dataset is already running."
    }


def _one_build_at_a_time(label: str):
    """
    Decorator for IndexBuilder.build(): refuse a second identical build while one is running.

    Two concurrent builds of the same index wrote the same files at the same time and corrupted them (a double
    click on "Build" is enough). The service runs one worker process, so a process-local lock is enough.
    """
    def decorator(build):
        @functools.wraps(build)
        def wrapper(self, dataset_name, *args, **kwargs):
            if not dataset_name:
                raise ValueError("dataset_name is required")
            with try_lock(f"{label}:{index_slug(dataset_name)}") as acquired:
                if not acquired:
                    return _already_running(label, dataset_name)
                return build(self, dataset_name, *args, **kwargs)
        return wrapper
    return decorator


def _one_build_all_at_a_time(build_all):
    """Same protection for build_all_indices(), which also writes one shared preprocessing file per dataset."""
    @functools.wraps(build_all)
    def wrapper(dataset_name, *args, **kwargs):
        if not dataset_name:
            raise ValueError("dataset_name is required")
        with try_lock(f"all:{index_slug(dataset_name)}") as acquired:
            if not acquired:
                return _already_running("build-all", dataset_name)
            return build_all(dataset_name, *args, **kwargs)
    return wrapper


def _check_dataset_in_database(dataset_name: str, table_name: str) -> None:
    """
    Check if dataset exists in database and raise helpful error if not.
    
    Args:
        dataset_name: Original dataset name (e.g., 'cranfield', 'beir/quora')
        table_name: Database table name (e.g., 'dataset_cranfield')
        
    Raises:
        FileNotFoundError: If dataset table doesn't exist in database
    """
    from common.database import get_db_connection
    
    with closing(get_db_connection()) as conn:  # `with conn` only ends the transaction, it never closed the connection
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name = %s
                )
            """, (table_name,))
            
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                # Get list of available datasets in database
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE 'dataset_%'
                    ORDER BY table_name
                """)
                available_tables = [row[0] for row in cur.fetchall()]
                known_ids = known_dataset_ids()
                available_datasets = [table_to_dataset_id(t, 'dataset', known_ids) for t in available_tables]
                
                # Build clear, structured error message
                error_parts = [
                    f"Dataset '{dataset_name}' is not uploaded to the database.",
                    "",
                    "Problem Details:",
                    f"  • Expected database table: {table_name}",
                    f"  • Table status: Does not exist",
                    f"  • Available datasets in database: {', '.join(available_datasets) if available_datasets else 'None'}",
                    "",
                    "Required Action:",
                    "  The dataset exists locally but must be uploaded to PostgreSQL before building indices.",
                    "",
                    "How to Upload:",
                    "  1. Navigate to the Dataset Management page",
                    f"  2. Locate dataset '{dataset_name}' in the downloaded list",
                    "  3. Click the 'Upload to Database' button",
                    "  4. Wait for upload completion (may take several minutes)",
                    "  5. Return to Index Management page to build indices",
                    "",
                    "Why This is Needed:",
                    "  Indices are built from preprocessed documents stored in PostgreSQL for efficient",
                    "  indexing and retrieval operations. Raw dataset files must first be processed",
                    "  and stored in the database."
                ]
                
                error_msg = "\n".join(error_parts)
                
                logger.error(f"Dataset not in database: {dataset_name} (table: {table_name})")
                raise FileNotFoundError(error_msg)


def _preprocess_documents(
    dataset_name: str, 
    preprocessing_config: Optional[dict] = None,
    shared_path: Optional[str] = None
) -> str:
    """
    Preprocess documents from database using the provided config.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        preprocessing_config: Preprocessing configuration dictionary
        shared_path: If provided, use this path instead of creating temporary file
        
    Returns:
        Path to the preprocessed documents file
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    try:
        from services.preprocessing.preprocessing_service import preprocess, PreprocessingConfig
    except ImportError:
        logger.error("Preprocessing service not available")
        raise ImportError("Preprocessing service is required but not available")
    
    logger.info(f"Preprocessing documents for dataset: {dataset_name}")
    
    if preprocessing_config:
        logger.info(f"Using custom preprocessing config: {preprocessing_config}")
        config = PreprocessingConfig(**preprocessing_config)
    else:
        logger.info("Using default preprocessing config")
        config = PreprocessingConfig()
    
    # Get the correct table name for this dataset
    table_prefix = get_config("database.table_prefix", "dataset")
    table_name = build_table_name(dataset_name, table_prefix)
    
    # Check if dataset exists in database first
    _check_dataset_in_database(dataset_name, table_name)
    
    logger.info(f"Reading documents from table: {table_name}")
    
    # Use shared path if provided, otherwise create temporary file
    if shared_path:
        processed_path = shared_path
        processed_file = open(processed_path, 'w', encoding='utf-8')
        logger.info(f"Using shared preprocessing file: {processed_path}")
    else:
        processed_file = NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl', encoding='utf-8')
        processed_path = processed_file.name
        logger.info(f"Using temporary preprocessing file: {processed_path}")
    
    try:
        from common.database import get_db_connection
        
        with closing(get_db_connection()) as conn:
            # Named (server-side) cursor: rows are streamed instead of loading the whole table into memory
            with conn.cursor(name="indexing_preprocess") as cur:
                cur.itersize = SERVER_CURSOR_ROWS
                # Read documents from dataset-specific table
                cur.execute(f"SELECT doc_id, text FROM {table_name} ORDER BY doc_id")
                
                doc_count = 0
                with jsonlines.Writer(processed_file) as writer:
                    for row in cur:
                        doc_id, text = row
                        
                        if not text or not text.strip():
                            logger.warning(f"Skipping empty document: {doc_id}")
                            continue
                        
                        tokens = preprocess(text, config)
                        
                        if not tokens:
                            logger.warning(f"No tokens after preprocessing for doc: {doc_id}")
                            continue
                        
                        writer.write({
                            "doc_id": doc_id,
                            "text": text,
                            "tokens": tokens
                        })
                        
                        doc_count += 1
                        
                        if doc_count % 10000 == 0:
                            logger.info(f"Preprocessed {doc_count} documents...")
        
        processed_file.close()  # jsonlines.Writer does not close a file object it was given: flush before indexing
        logger.info(f"Preprocessing complete: {doc_count} documents written to {processed_path}")
        return processed_path
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        processed_file.close()
        Path(processed_path).unlink(missing_ok=True)
        raise


class IndexBuilder:
    """Base class for index building operations."""
    
    @staticmethod
    def _handle_existing_index(
        indexer, 
        force_rebuild: bool, 
        index_type: str
    ) -> Dict[str, str]:
        """
        Check if index exists and handle based on force_rebuild flag.
        
        Args:
            indexer: Indexer instance to check
            force_rebuild: Whether to rebuild existing index
            index_type: Name of index type for logging
            
        Returns:
            Skip message dictionary if index exists and not forcing rebuild,
            None otherwise to continue building
        """
        if indexer.exists() and not force_rebuild:
            logger.info(f"{index_type} index already exists, skipping build")
            return {
                "status": "skipped",
                "message": f"{index_type} index already exists. Use force_rebuild=true to rebuild."
            }
        return None


class TFIDFIndexBuilder(IndexBuilder):
    """Builds TF-IDF index from documents."""
    
    @_one_build_at_a_time("TF-IDF")
    def build(
        self, 
        dataset_name: str, 
        force_rebuild: bool = False, 
        preprocessing_config: Optional[dict] = None,
        preprocessed_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build TF-IDF index.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
            force_rebuild: Force rebuild even if index exists
            preprocessing_config: Custom preprocessing configuration
            preprocessed_path: Path to pre-built preprocessed file (skips preprocessing)
            
        Returns:
            Dictionary with build status and index statistics
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        logger.info(f"Building TF-IDF index for dataset: {dataset_name} (force_rebuild={force_rebuild})")
        
        indexer = TFIDFIndexer(dataset_name=dataset_name)
        
        skip_result = self._handle_existing_index(indexer, force_rebuild, "TF-IDF")
        if skip_result:
            return skip_result
        
        should_cleanup = False
        try:
            # Use provided preprocessed file or create new one
            if preprocessed_path:
                logger.info(f"Using existing preprocessed file: {preprocessed_path}")
                processed_path = preprocessed_path
            else:
                processed_path = _preprocess_documents(dataset_name, preprocessing_config)
                should_cleanup = True
            
            indexer.build(processed_path=processed_path)
            
            indexer.save()
            
            logger.info("TF-IDF index built successfully")
            
            # Get file sizes
            vectorizer_path = indexer.vectorizer_path
            matrix_path = indexer.matrix_path
            
            total_size_mb = 0
            if vectorizer_path.exists():
                total_size_mb += vectorizer_path.stat().st_size / (1024**2)
            if matrix_path.exists():
                total_size_mb += matrix_path.stat().st_size / (1024**2)
            
            return {
                "status": "success",
                "vocab_size": len(indexer.vectorizer.vocabulary_),
                "matrix_shape": list(indexer.matrix.shape),
                "total_docs": len(indexer.doc_ids),
                "index_size_mb": round(total_size_mb, 2),
                "preprocessing_applied": preprocessing_config is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to build TF-IDF index: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            if should_cleanup and processed_path:
                Path(processed_path).unlink(missing_ok=True)


class EmbeddingIndexBuilder(IndexBuilder):
    """Builds embedding-based index from documents."""
    
    @_one_build_at_a_time("Embedding")
    def build(
        self, 
        dataset_name: str, 
        force_rebuild: bool = False, 
        preprocessing_config: Optional[dict] = None,
        preprocessed_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build Embedding index.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
            force_rebuild: Force rebuild even if index exists
            preprocessing_config: Custom preprocessing configuration
            preprocessed_path: Path to pre-built preprocessed file (skips preprocessing)
            
        Returns:
            Dictionary with build status and index statistics
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        logger.info(f"Building Embedding index for dataset: {dataset_name} (force_rebuild={force_rebuild})")
        
        indexer = EmbeddingIndexer(dataset_name=dataset_name)
        
        skip_result = self._handle_existing_index(indexer, force_rebuild, "Embedding")
        if skip_result:
            return skip_result
        
        should_cleanup = False
        try:
            # Use provided preprocessed file or create new one
            if preprocessed_path:
                logger.info(f"Using existing preprocessed file: {preprocessed_path}")
                processed_path = preprocessed_path
            else:
                processed_path = _preprocess_documents(dataset_name, preprocessing_config)
                should_cleanup = True
            
            indexer.build(processed_path=processed_path)
            
            indexer.save()
            
            logger.info("Embedding index built successfully")
            
            # Get file sizes
            embeddings_path = indexer.embeddings_path
            doc_map_path = indexer.doc_id_map_path
            
            total_size_mb = 0
            if embeddings_path.exists():
                total_size_mb += embeddings_path.stat().st_size / (1024**2)
            if doc_map_path.exists():
                total_size_mb += doc_map_path.stat().st_size / (1024**2)
            
            return {
                "status": "success",
                "shape": list(indexer.embeddings.shape),
                "total_docs": len(indexer.doc_ids),
                "index_size_mb": round(total_size_mb, 2),
                "memory_mb": round(indexer.embeddings.nbytes / (1024**2), 2),
                "device": indexer.device,
                "preprocessing_applied": preprocessing_config is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to build Embedding index: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            if should_cleanup and processed_path:
                Path(processed_path).unlink(missing_ok=True)


class BM25IndexBuilder(IndexBuilder):
    """Builds BM25 index from documents."""
    
    @_one_build_at_a_time("BM25")
    def build(
        self, 
        dataset_name: str, 
        force_rebuild: bool = False, 
        preprocessing_config: Optional[dict] = None,
        preprocessed_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build BM25 index.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
            force_rebuild: Force rebuild even if index exists
            preprocessing_config: Custom preprocessing configuration
            preprocessed_path: Path to pre-built preprocessed file (skips preprocessing)
            
        Returns:
            Dictionary with build status and index statistics
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        logger.info(f"Building BM25 index for dataset: {dataset_name} (force_rebuild={force_rebuild})")
        
        indexer = BM25Indexer(dataset_name=dataset_name)
        
        skip_result = self._handle_existing_index(indexer, force_rebuild, "BM25")
        if skip_result:
            return skip_result
        
        should_cleanup = False
        try:
            # Use provided preprocessed file or create new one
            if preprocessed_path:
                logger.info(f"Using existing preprocessed file: {preprocessed_path}")
                processed_path = preprocessed_path
            else:
                processed_path = _preprocess_documents(dataset_name, preprocessing_config)
                should_cleanup = True
            
            indexer.build(processed_path=processed_path)
            
            indexer.save()
            
            logger.info("BM25 index built successfully")
            
            # Get file size
            index_path = indexer.index_path
            index_size_mb = 0
            if index_path.exists():
                index_size_mb = index_path.stat().st_size / (1024**2)
            
            return {
                "status": "success",
                "vocab_size": len(indexer.inverted_index),
                "total_docs": indexer.total_docs,
                "avg_doc_length": round(indexer.avg_doc_length, 2),
                "index_size_mb": round(index_size_mb, 2),
                "preprocessing_applied": preprocessing_config is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            if should_cleanup and processed_path:
                Path(processed_path).unlink(missing_ok=True)


def build_tfidf_index(dataset_name: str, force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Public API: Build TF-IDF index.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        force_rebuild: Force rebuild even if index exists
        preprocessing_config: Custom preprocessing configuration
        
    Returns:
        Dictionary with build status and statistics
    """
    builder = TFIDFIndexBuilder()
    return builder.build(dataset_name, force_rebuild, preprocessing_config)


def build_embedding_index(dataset_name: str, force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Public API: Build Embedding index.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        force_rebuild: Force rebuild even if index exists
        preprocessing_config: Custom preprocessing configuration
        
    Returns:
        Dictionary with build status and statistics
    """
    builder = EmbeddingIndexBuilder()
    return builder.build(dataset_name, force_rebuild, preprocessing_config)


def build_bm25_index(dataset_name: str, force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Public API: Build BM25 index.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        force_rebuild: Force rebuild even if index exists
        preprocessing_config: Custom preprocessing configuration
        
    Returns:
        Dictionary with build status and statistics
    """
    builder = BM25IndexBuilder()
    return builder.build(dataset_name, force_rebuild, preprocessing_config)


@_one_build_all_at_a_time
def build_all_indices(dataset_name: str, force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Public API: Build all indices (TF-IDF, BM25, Embedding) with shared preprocessing.
    
    This function builds preprocessing once and reuses it for all three indices,
    significantly reducing build time.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        force_rebuild: Force rebuild even if indices exist
        preprocessing_config: Custom preprocessing configuration
        
    Returns:
        Dictionary with build status for all indices
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    logger.info(f"Building all indices for dataset: {dataset_name} with shared preprocessing")
    
    results = {
        "dataset": dataset_name,
        "preprocessing_shared": True,
        "indices": {}
    }
    
    # Step 1: Build preprocessing once and save to shared location
    processed_data_dir = Path(get_config("paths.processed", "data/processed"))
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    safe_dataset_name = table_slug(dataset_name)
    shared_preprocessed_path = processed_data_dir / f"{safe_dataset_name}_shared.jsonl"
    
    try:
        if not force_rebuild and all(
            indexer_class(dataset_name=dataset_name).exists()
            for indexer_class in (TFIDFIndexer, BM25Indexer, EmbeddingIndexer)
        ):
            # Every index exists and no rebuild was requested: skip the expensive preprocessing pass (it used to run
            # for minutes and its result was thrown away). The "is the dataset uploaded?" check is kept.
            logger.info("All indices already exist - skipping the shared preprocessing")
            _check_dataset_in_database(dataset_name, build_table_name(dataset_name, configured_table_prefix()))
        else:
            logger.info("Step 1/4: Building shared preprocessing file...")
            _preprocess_documents(
                dataset_name=dataset_name,
                preprocessing_config=preprocessing_config,
                shared_path=str(shared_preprocessed_path)
            )
            logger.info(f"Shared preprocessing complete: {shared_preprocessed_path}")
        
        # Step 2: Build TF-IDF using shared file
        logger.info("Step 2/4: Building TF-IDF index...")
        tfidf_builder = TFIDFIndexBuilder()
        results["indices"]["tfidf"] = tfidf_builder.build(
            dataset_name=dataset_name,
            force_rebuild=force_rebuild,
            preprocessing_config=preprocessing_config,
            preprocessed_path=str(shared_preprocessed_path)
        )
        
        # Step 3: Build BM25 using shared file
        logger.info("Step 3/4: Building BM25 index...")
        bm25_builder = BM25IndexBuilder()
        results["indices"]["bm25"] = bm25_builder.build(
            dataset_name=dataset_name,
            force_rebuild=force_rebuild,
            preprocessing_config=preprocessing_config,
            preprocessed_path=str(shared_preprocessed_path)
        )
        
        # Step 4: Build Embedding using shared file
        logger.info("Step 4/4: Building Embedding index...")
        embedding_builder = EmbeddingIndexBuilder()
        results["indices"]["embedding"] = embedding_builder.build(
            dataset_name=dataset_name,
            force_rebuild=force_rebuild,
            preprocessing_config=preprocessing_config,
            preprocessed_path=str(shared_preprocessed_path)
        )
        
        logger.info("All indices built successfully with shared preprocessing")
        results["status"] = "success"
        
        # Count successful builds
        success_count = sum(1 for idx_result in results["indices"].values() 
                          if idx_result.get("status") in ("success", "skipped"))
        results["success_count"] = success_count
        results["total_count"] = 3
        
    except Exception as e:
        logger.error(f"Failed to build all indices: {e}", exc_info=True)
        results["status"] = "error"
        results["message"] = str(e)
    
    finally:
        # Clean up shared preprocessing file
        if shared_preprocessed_path.exists():
            logger.info(f"Cleaning up shared preprocessing file: {shared_preprocessed_path}")
            shared_preprocessed_path.unlink(missing_ok=True)
    
    return results


__all__ = [
    "build_tfidf_index",
    "build_embedding_index",
    "build_bm25_index",
    "build_all_indices",
    "TFIDFIndexBuilder",
    "EmbeddingIndexBuilder",
    "BM25IndexBuilder",
]
