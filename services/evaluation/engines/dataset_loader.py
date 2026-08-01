"""
Dataset loader for evaluation.
Loads queries and qrels from ir_datasets.
"""

import ir_datasets
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load queries and qrels from ir_datasets for evaluation"""
    
    @staticmethod
    def load_queries_and_qrels(dataset_name: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
        """
        Load queries and qrels from ir_datasets.
        
        Args:
            dataset_name: Dataset identifier (e.g., 'beir/quora/test')
            
        Returns:
            Tuple of (queries, qrels) where:
                - queries: Dict mapping query_id -> query_text
                - qrels: Dict mapping query_id -> (doc_id -> relevance)
                
        Raises:
            ValueError: If dataset doesn't have qrels
        """
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = ir_datasets.load(dataset_name)
        
        queries = {}
        for query in dataset.queries_iter():
            queries[query.query_id] = query.text
        
        qrels = {}
        if not hasattr(dataset, 'qrels_iter'):
            raise ValueError(f"Dataset {dataset_name} does not have qrels. Cannot evaluate.")
        
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        logger.info(f"Loaded {len(queries)} queries and {len(qrels)} qrels")
        return queries, qrels
    
    @staticmethod
    def validate_dataset(dataset_name: str) -> bool:
        """
        Validate that dataset exists and has required features.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            dataset = ir_datasets.load(dataset_name)
            
            has_queries = hasattr(dataset, 'queries_iter')
            has_qrels = hasattr(dataset, 'qrels_iter')
            
            if not has_queries:
                logger.error(f"Dataset {dataset_name} has no queries")
                return False
            
            if not has_qrels:
                logger.error(f"Dataset {dataset_name} has no qrels")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate dataset {dataset_name}: {e}")
            return False
