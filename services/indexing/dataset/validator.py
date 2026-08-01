"""
Dataset validation functionality.
"""

import logging
from typing import Dict, Tuple

import ir_datasets

from common.exceptions import DatasetValidationError


logger = logging.getLogger(__name__)


class DatasetValidator:
    """
    Validates and checks IR datasets for required features.
    
    Implements validation logic for dataset existence and required components
    (queries and qrels) needed for evaluation in the IR system.
    """
    
    def __init__(self, dataset_id: str):
        """
        Initialize validator for a specific dataset.
        
        Args:
            dataset_id: Identifier of the dataset to validate
        """
        self.dataset_id = dataset_id
        self._validation_cache: Dict[str, bool] = {}
    
    def validate_exists(self) -> None:
        """
        Validate that dataset exists in ir_datasets registry.
        
        Raises:
            DatasetValidationError: If dataset not found in registry
        """
        try:
            ir_datasets.load(self.dataset_id)
            logger.debug(f"Dataset '{self.dataset_id}' found in registry")
        except KeyError as e:
            # Provide helpful suggestions for common mistakes
            suggestions = []
            dataset_lower = self.dataset_id.lower()
            
            # Check for nano-beir datasets (not available in 0.5.9)
            if 'nano-beir' in dataset_lower:
                base_name = self.dataset_id.replace('nano-beir/', '')
                suggestions.append(f"Try the full-size version instead: 'beir/{base_name}' or '{base_name}'")
            
            # Check for missing /test suffix
            if '/test' not in self.dataset_id and 'beir' in dataset_lower:
                suggestions.append(f"Try adding '/test' suffix: '{self.dataset_id}/test'")
            
            error_msg = (
                f"Dataset '{self.dataset_id}' not found in ir_datasets registry "
                f"(installed version {ir_datasets.__version__}). "
                "Check the dataset ID or upgrade ir_datasets."
            )
            
            if suggestions:
                error_msg += "\n\nSuggestions:\n" + "\n".join(f"  • {s}" for s in suggestions)
            
            logger.error(error_msg)
            raise DatasetValidationError(error_msg) from e
    
    def validate_has_required_features(self) -> None:
        """
        Validate that dataset has queries and qrels needed for evaluation.
        
        According to project requirements, datasets must have >200K documents
        and include testing data with qrels for evaluation.
        
        Raises:
            DatasetValidationError: If queries or qrels are missing
        """
        has_queries, has_qrels = self._check_features()
        
        if not has_queries:
            error_msg = f"Dataset '{self.dataset_id}' has no queries - required for evaluation."
            logger.error(error_msg)
            raise DatasetValidationError(error_msg)
        
        if not has_qrels:
            error_msg = f"Dataset '{self.dataset_id}' has no qrels - required for evaluation."
            logger.error(error_msg)
            raise DatasetValidationError(error_msg)
        
        logger.info(f"Dataset '{self.dataset_id}' has all required features")
    
    def _check_features(self) -> Tuple[bool, bool]:
        """
        Check if dataset has queries and qrels.
        
        Tries both /test variant and base dataset to handle different
        dataset structures in ir_datasets library.
        
        Returns:
            Tuple of (has_queries, has_qrels) boolean flags
        """
        has_queries = False
        has_qrels = False
        
        candidates = [f"{self.dataset_id}/test", self.dataset_id]
        
        for candidate in candidates:
            try:
                ds = ir_datasets.load(candidate)
                
                if hasattr(ds, "queries_iter"):
                    has_queries = True
                    logger.debug(f"Found queries in '{candidate}'")
                
                if hasattr(ds, "qrels_iter"):
                    has_qrels = True
                    logger.debug(f"Found qrels in '{candidate}'")
                
                if has_queries and has_qrels:
                    break
                    
            except Exception as e:
                logger.debug(f"Could not load candidate '{candidate}': {e}")
                continue
        
        return has_queries, has_qrels
