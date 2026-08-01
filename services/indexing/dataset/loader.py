"""
Dataset loading with retry logic.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

import ir_datasets

from common.exceptions import DatasetValidationError
from services.indexing.dataset.constants import (
    DEFAULT_VALIDATION_RETRIES,
    DEFAULT_RETRY_DELAY
)


logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and retrieves statistics from IR datasets.
    
    Implements retry logic for robust dataset loading and extracts
    comprehensive statistics for indexing service.
    """
    
    def __init__(self, dataset_id: str):
        """
        Initialize loader for a specific dataset.
        
        Args:
            dataset_id: Identifier of the dataset to load
        """
        self.dataset_id = dataset_id
    
    def load_with_retry(
        self, 
        max_retries: int = DEFAULT_VALIDATION_RETRIES,
        retry_delay: int = DEFAULT_RETRY_DELAY
    ) -> Dict[str, Any]:
        """
        Load dataset with exponential backoff retry logic.
        
        Implements robust loading with retries for handling temporary
        network issues or dataset download interruptions.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay in seconds between retries
            
        Returns:
            Dictionary containing dataset statistics
            
        Raises:
            DatasetValidationError: If loading fails after all retries
        """
        last_exception: Optional[Exception] = None
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Loading dataset attempt {attempt}/{max_retries}")
                return self._load_and_extract_stats()
                
            except DatasetValidationError:
                raise
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Dataset load attempt {attempt} failed: {e}",
                    exc_info=attempt == max_retries
                )
                
                if attempt < max_retries:
                    # Exponential backoff
                    current_delay = retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
        
        error_msg = f"Failed to load dataset after {max_retries} retries"
        logger.error(error_msg)
        raise DatasetValidationError(error_msg) from last_exception
    
    def _load_and_extract_stats(self) -> Dict[str, Any]:
        """
        Load dataset and extract comprehensive statistics.
        
        Returns:
            Dictionary with dataset statistics and metadata
        """
        docs_ds = ir_datasets.load(self.dataset_id)
        test_ds = self._load_test_variant(docs_ds)
        
        stats = self._extract_statistics(docs_ds, test_ds)
        
        logger.info(
            f"Dataset stats extracted: docs={stats['doc_count']}, "
            f"queries={stats['query_count']}, qrels={stats['qrels_count']}"
        )
        
        return stats
    
    def _load_test_variant(self, fallback_ds):
        """
        Try to load test variant of dataset, fallback to base.
        
        Args:
            fallback_ds: Fallback dataset if test variant doesn't exist
            
        Returns:
            Test dataset or fallback dataset
        """
        try:
            return ir_datasets.load(f"{self.dataset_id}/test")
        except Exception as e:
            logger.debug(f"Test variant not found, using base dataset: {e}")
            return fallback_ds
    
    def _extract_statistics(self, docs_ds, test_ds) -> Dict[str, Any]:
        """
        Extract statistics from dataset objects.
        
        Args:
            docs_ds: Documents dataset
            test_ds: Test dataset (queries and qrels)
            
        Returns:
            Dictionary with extracted statistics
        """
        # Force download by iterating first document
        # This ensures the dataset files are actually downloaded to disk
        if hasattr(docs_ds, 'docs_iter'):
            try:
                logger.info("Forcing dataset download by reading first document...")
                first_doc = next(iter(docs_ds.docs_iter()))
                logger.info(f"First document loaded: {first_doc.doc_id[:50] if hasattr(first_doc, 'doc_id') else 'N/A'}")
            except Exception as e:
                logger.warning(f"Could not iterate first document (may be lazy): {e}")
        
        doc_count = self._safe_count(docs_ds, "docs_count")
        query_count = self._safe_count(test_ds, "queries_count")
        qrels_count = self._safe_count(test_ds, "qrels_count")
        
        return {
            "dataset_name": self.dataset_id,
            "doc_count": doc_count,
            "query_count": query_count,
            "qrels_count": qrels_count,
            "timestamp": datetime.utcnow().isoformat(),
            "validation_status": "success",
            "ir_datasets_version": ir_datasets.__version__
        }
    
    @staticmethod
    def _safe_count(dataset, count_method: str) -> int:
        """
        Safely get count from dataset, return 0 if not available.
        
        Args:
            dataset: Dataset object
            count_method: Name of the count method to call
            
        Returns:
            Count value or 0 if not available
        """
        try:
            if hasattr(dataset, count_method):
                count = getattr(dataset, count_method)()
                return count if count is not None else 0
            return 0
        except Exception as e:
            logger.warning(f"Could not get {count_method}: {e}")
            return 0
