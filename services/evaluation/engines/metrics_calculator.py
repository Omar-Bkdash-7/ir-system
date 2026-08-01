"""
Metrics calculation engine for IR evaluation.
Implements standard metrics: MAP, nDCG, Recall, Precision.
"""

import numpy as np
from typing import List, Dict


class MetricsCalculator:
    """Calculate IR metrics for evaluation"""
    
    @staticmethod
    def calculate_ndcg(retrieved_doc_ids: List[str], 
                      relevant_docs: Dict[str, int], 
                      k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG@k).
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_docs: Dictionary mapping doc_id -> relevance score
            k: Cutoff for evaluation
            
        Returns:
            nDCG@k score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids[:k]):
            if doc_id in relevant_docs:
                rel = relevant_docs[doc_id]
                dcg += (2**rel - 1) / np.log2(i + 2)
        
        ideal_rels = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def calculate_average_precision(retrieved_doc_ids: List[str], 
                                    relevant_docs: Dict[str, int]) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_docs: Dictionary mapping doc_id -> relevance score
            
        Returns:
            Average Precision score
        """
        num_relevant = 0
        sum_precision = 0.0
        
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i
        
        total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
        return sum_precision / total_relevant if total_relevant > 0 else 0.0
    
    @staticmethod
    def calculate_recall(retrieved_doc_ids: List[str], 
                        relevant_docs: Dict[str, int]) -> float:
        """
        Calculate Recall@k.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_docs: Dictionary mapping doc_id -> relevance score
            
        Returns:
            Recall score
        """
        relevant_retrieved = sum(
            1 for doc_id in retrieved_doc_ids 
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0
        )
        total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
        
        return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    
    @staticmethod
    def calculate_precision(retrieved_doc_ids: List[str], 
                           relevant_docs: Dict[str, int]) -> float:
        """
        Calculate Precision@k.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_docs: Dictionary mapping doc_id -> relevance score
            
        Returns:
            Precision score
        """
        relevant_retrieved = sum(
            1 for doc_id in retrieved_doc_ids 
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0
        )
        
        return relevant_retrieved / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
    
    @classmethod
    def calculate_all_metrics(cls, 
                             retrieved_doc_ids: List[str], 
                             relevant_docs: Dict[str, int], 
                             k: int) -> Dict[str, float]:
        """
        Calculate all metrics for a single query.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_docs: Dictionary mapping doc_id -> relevance score
            k: Cutoff for evaluation
            
        Returns:
            Dictionary with all metric scores
        """
        return {
            "ndcg": cls.calculate_ndcg(retrieved_doc_ids, relevant_docs, k),
            "ap": cls.calculate_average_precision(retrieved_doc_ids, relevant_docs),
            "recall": cls.calculate_recall(retrieved_doc_ids, relevant_docs),
            "precision": cls.calculate_precision(retrieved_doc_ids, relevant_docs)
        }
