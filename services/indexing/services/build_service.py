"""
Index building service.

Provides functionality for building TF-IDF, Embedding, and BM25 indices
with force rebuild capability, preprocessing integration, and error handling.
"""

import logging
import jsonlines
from pathlib import Path
from typing import Dict, Any, Optional
from tempfile import NamedTemporaryFile

from services.indexing.indexers import TFIDFIndexer, EmbeddingIndexer, BM25Indexer
from services.indexing.dataset_service import get_metadata
from common.config import get_config


logger = logging.getLogger(__name__)


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
    
    with get_db_connection() as conn:
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
                available_datasets = [t.replace('dataset_', '').replace('_', '/') for t in available_tables]
                
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


def _preprocess_documents(preprocessing_config: Optional[dict] = None) -> str:
    """
    Preprocess documents from database using the provided config.
    
    Args:
        preprocessing_config: Preprocessing configuration dictionary
        
    Returns:
        Path to the preprocessed documents file
    """
    try:
        from services.preprocessing.preprocessing_service import preprocess, PreprocessingConfig
    except ImportError:
        logger.error("Preprocessing service not available")
        raise ImportError("Preprocessing service is required but not available")
    
    metadata = get_metadata()
    dataset_name = metadata.get("dataset_name", "unknown")
    
    logger.info(f"Preprocessing documents for dataset: {dataset_name}")
    
    if preprocessing_config:
        logger.info(f"Using custom preprocessing config: {preprocessing_config}")
        config = PreprocessingConfig(**preprocessing_config)
    else:
        logger.info("Using default preprocessing config")
        config = PreprocessingConfig()
    
    # Get the correct table name for this dataset
    table_prefix = get_config("database.table_prefix", "dataset")
    safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_').replace('.', '_')
    table_name = f"{table_prefix}_{safe_dataset_name}"
    
    # Check if dataset exists in database first
    _check_dataset_in_database(dataset_name, table_name)
    
    logger.info(f"Reading documents from table: {table_name}")
    
    processed_file = NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl', encoding='utf-8')
    processed_path = processed_file.name
    
    try:
        from common.database import get_db_connection
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
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
        
        logger.info(f"Preprocessing complete: {doc_count} documents written to {processed_path}")
        return processed_path
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
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
    
    def build(self, force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
        """
        Build TF-IDF index.
        
        Args:
            force_rebuild: Force rebuild even if index exists
            preprocessing_config: Custom preprocessing configuration
            
        Returns:
            Dictionary with build status and index statistics
        """
        logger.info(f"Building TF-IDF index (force_rebuild={force_rebuild})")
        
        # Get dataset name from metadata
        metadata = get_metadata()
        dataset_name = metadata.get("dataset_name")
        
        indexer = TFIDFIndexer(dataset_name=dataset_name)
        
        skip_result = self._handle_existing_index(indexer, force_rebuild, "TF-IDF")
        if skip_result:
            return skip_result
        
        processed_path = None
        try:
            # Always preprocess documents from database
            processed_path = _preprocess_documents(preprocessing_config)
            indexer.build(processed_path=processed_path)
            
            indexer.save()
            
            logger.info("TF-IDF index built successfully")
            
            # Get file sizes
            vectorizer_path = Path(get_config("indices.tfidf_vectorizer", "data/indices/tfidf_vectorizer.pkl"))
            matrix_path = Path(get_config("indices.tfidf_matrix", "data/indices/tfidf_matrix.npz"))
            
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
            if processed_path:
                Path(processed_path).unlink(missing_ok=True)


class EmbeddingIndexBuilder(IndexBuilder):
    """Builds embedding-based index from documents."""
    
    def build(self, force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
        """
        Build Embedding index.
        
        Args:
            force_rebuild: Force rebuild even if index exists
            preprocessing_config: Custom preprocessing configuration
            
        Returns:
            Dictionary with build status and index statistics
        """
        logger.info(f"Building Embedding index (force_rebuild={force_rebuild})")
        
        # Get dataset name from metadata
        metadata = get_metadata()
        dataset_name = metadata.get("dataset_name")
        
        indexer = EmbeddingIndexer(dataset_name=dataset_name)
        
        skip_result = self._handle_existing_index(indexer, force_rebuild, "Embedding")
        if skip_result:
            return skip_result
        
        processed_path = None
        try:
            # Always preprocess documents from database
            processed_path = _preprocess_documents(preprocessing_config)
            indexer.build(processed_path=processed_path)
            
            indexer.save()
            
            logger.info("Embedding index built successfully")
            
            # Get file sizes
            embeddings_path = Path(get_config("indices.embeddings", "data/indices/embeddings.npy"))
            doc_map_path = Path(get_config("indices.doc_id_map", "data/indices/doc_id_map.json"))
            
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
            if processed_path:
                Path(processed_path).unlink(missing_ok=True)


class BM25IndexBuilder(IndexBuilder):
    """Builds BM25 index from documents."""
    
    def build(self, force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
        """
        Build BM25 index.
        
        Args:
            force_rebuild: Force rebuild even if index exists
            preprocessing_config: Custom preprocessing configuration
            
        Returns:
            Dictionary with build status and index statistics
        """
        logger.info(f"Building BM25 index (force_rebuild={force_rebuild})")
        
        # Get dataset name from metadata
        metadata = get_metadata()
        dataset_name = metadata.get("dataset_name")
        
        indexer = BM25Indexer(dataset_name=dataset_name)
        
        skip_result = self._handle_existing_index(indexer, force_rebuild, "BM25")
        if skip_result:
            return skip_result
        
        processed_path = None
        try:
            # Always preprocess documents from database
            processed_path = _preprocess_documents(preprocessing_config)
            indexer.build(processed_path=processed_path)
            
            indexer.save()
            
            logger.info("BM25 index built successfully")
            
            # Get file size
            index_path = Path(get_config("indices.bm25_index", "data/indices/bm25_index.pkl"))
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
            if processed_path:
                Path(processed_path).unlink(missing_ok=True)


def build_tfidf_index(force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Public API: Build TF-IDF index.
    
    Args:
        force_rebuild: Force rebuild even if index exists
        preprocessing_config: Custom preprocessing configuration
        
    Returns:
        Dictionary with build status and statistics
    """
    builder = TFIDFIndexBuilder()
    return builder.build(force_rebuild, preprocessing_config)


def build_embedding_index(force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Public API: Build Embedding index.
    
    Args:
        force_rebuild: Force rebuild even if index exists
        preprocessing_config: Custom preprocessing configuration
        
    Returns:
        Dictionary with build status and statistics
    """
    builder = EmbeddingIndexBuilder()
    return builder.build(force_rebuild, preprocessing_config)


def build_bm25_index(force_rebuild: bool = False, preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Public API: Build BM25 index.
    
    Args:
        force_rebuild: Force rebuild even if index exists
        preprocessing_config: Custom preprocessing configuration
        
    Returns:
        Dictionary with build status and statistics
    """
    builder = BM25IndexBuilder()
    return builder.build(force_rebuild, preprocessing_config)


__all__ = [
    "build_tfidf_index",
    "build_embedding_index",
    "build_bm25_index",
    "TFIDFIndexBuilder",
    "EmbeddingIndexBuilder",
    "BM25IndexBuilder",
]
