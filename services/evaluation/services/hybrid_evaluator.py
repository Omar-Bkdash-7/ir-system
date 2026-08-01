"""
Hybrid retrieval evaluator service.
Handles evaluation of hybrid retrieval models (serial and parallel).
"""

import requests
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from common.config import get_config
from services.evaluation.engines.metrics_calculator import MetricsCalculator
from services.evaluation.engines.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class HybridEvaluator:
    """Evaluate hybrid retrieval models"""
    
    def __init__(self):
        self.retrieval_service_url = get_config(
            "services.retrieval_url",
            "http://retrieval:8003"
        )
        self.preprocessing_service_url = get_config(
            "services.preprocessing_url",
            "http://preprocessing:8001"
        )
        self.metrics_calculator = MetricsCalculator()
        self.dataset_loader = DatasetLoader()
    
    def evaluate_hybrid_serial(
        self,
        dataset_name: str,
        first_model: str = "tfidf",
        second_model: str = "embedding",
        top_k: int = 100,
        query_limit: Optional[int] = None
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate hybrid serial retrieval.
        
        Args:
            dataset_name: Dataset identifier
            first_model: First model to use
            second_model: Second model to use
            top_k: Number of documents to retrieve
            query_limit: Limit number of queries
            
        Returns:
            Tuple of (results, metrics)
        """
        logger.info(f"Evaluating hybrid serial: {first_model} -> {second_model}")
        
        queries, qrels = self.dataset_loader.load_queries_and_qrels(dataset_name)
        
        if query_limit:
            queries = dict(list(queries.items())[:query_limit])
        
        results = {}
        all_ndcg = []
        all_ap = []
        all_recall = []
        all_precision = []
        
        for query_id, query_text in tqdm(queries.items(), desc="Evaluating hybrid serial"):
            if query_id not in qrels or not qrels[query_id]:
                continue
            
            try:
                query_tokens = self._preprocess(query_text)
                
                payload = {
                    "query_text": query_text,
                    "query_tokens": query_tokens,
                    "first_model": first_model,
                    "second_model": second_model,
                    "top_k": top_k,
                    "include_text": False
                }
                
                response = requests.post(
                    f"{self.retrieval_service_url}/retrieve/hybrid/serial",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                retrieved_docs = response.json().get("results", [])
                
                if not retrieved_docs:
                    continue
                
                retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_docs]
                relevant_docs = qrels[query_id]
                
                metrics = self.metrics_calculator.calculate_all_metrics(
                    retrieved_doc_ids, relevant_docs, top_k
                )
                
                all_ndcg.append(metrics["ndcg"])
                all_ap.append(metrics["ap"])
                all_recall.append(metrics["recall"])
                all_precision.append(metrics["precision"])
                
                num_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
                num_relevant_retrieved = sum(
                    1 for doc_id in retrieved_doc_ids 
                    if doc_id in relevant_docs and relevant_docs[doc_id] > 0
                )
                
                results[query_id] = {
                    "query_text": query_text,
                    "retrieved": retrieved_docs,
                    "num_retrieved": len(retrieved_docs),
                    "num_relevant": num_relevant,
                    "num_relevant_retrieved": num_relevant_retrieved,
                    "ap": metrics["ap"],
                    "ndcg": metrics["ndcg"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"]
                }
                
            except Exception as e:
                logger.error(f"Error evaluating query {query_id}: {e}", exc_info=True)
                continue
        
        overall_metrics = {
            "map": float(sum(all_ap) / len(all_ap)) if all_ap else 0.0,
            "ndcg": float(sum(all_ndcg) / len(all_ndcg)) if all_ndcg else 0.0,
            "recall": float(sum(all_recall) / len(all_recall)) if all_recall else 0.0,
            "precision": float(sum(all_precision) / len(all_precision)) if all_precision else 0.0,
            "num_queries": len(results)
        }
        
        return results, overall_metrics
    
    def evaluate_hybrid_parallel(
        self,
        dataset_name: str,
        models: List[str],
        fusion_method: str = "rrf",
        top_k: int = 100,
        query_limit: Optional[int] = None
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate hybrid parallel retrieval.
        
        Args:
            dataset_name: Dataset identifier
            models: List of models to use
            fusion_method: Fusion method ('rrf' or 'weighted')
            top_k: Number of documents to retrieve
            query_limit: Limit number of queries
            
        Returns:
            Tuple of (results, metrics)
        """
        logger.info(f"Evaluating hybrid parallel: {models} with {fusion_method}")
        
        queries, qrels = self.dataset_loader.load_queries_and_qrels(dataset_name)
        
        if query_limit:
            queries = dict(list(queries.items())[:query_limit])
        
        results = {}
        all_ndcg = []
        all_ap = []
        all_recall = []
        all_precision = []
        
        for query_id, query_text in tqdm(queries.items(), desc="Evaluating hybrid parallel"):
            if query_id not in qrels or not qrels[query_id]:
                continue
            
            try:
                query_tokens = self._preprocess(query_text)
                
                payload = {
                    "query_text": query_text,
                    "query_tokens": query_tokens,
                    "models": models,
                    "fusion_method": fusion_method,
                    "top_k": top_k,
                    "include_text": False
                }
                
                response = requests.post(
                    f"{self.retrieval_service_url}/retrieve/hybrid/parallel",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                retrieved_docs = response.json().get("results", [])
                
                if not retrieved_docs:
                    continue
                
                retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_docs]
                relevant_docs = qrels[query_id]
                
                metrics = self.metrics_calculator.calculate_all_metrics(
                    retrieved_doc_ids, relevant_docs, top_k
                )
                
                all_ndcg.append(metrics["ndcg"])
                all_ap.append(metrics["ap"])
                all_recall.append(metrics["recall"])
                all_precision.append(metrics["precision"])
                
                num_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
                num_relevant_retrieved = sum(
                    1 for doc_id in retrieved_doc_ids 
                    if doc_id in relevant_docs and relevant_docs[doc_id] > 0
                )
                
                results[query_id] = {
                    "query_text": query_text,
                    "retrieved": retrieved_docs,
                    "num_retrieved": len(retrieved_docs),
                    "num_relevant": num_relevant,
                    "num_relevant_retrieved": num_relevant_retrieved,
                    "ap": metrics["ap"],
                    "ndcg": metrics["ndcg"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"]
                }
                
            except Exception as e:
                logger.error(f"Error evaluating query {query_id}: {e}", exc_info=True)
                continue
        
        overall_metrics = {
            "map": float(sum(all_ap) / len(all_ap)) if all_ap else 0.0,
            "ndcg": float(sum(all_ndcg) / len(all_ndcg)) if all_ndcg else 0.0,
            "recall": float(sum(all_recall) / len(all_recall)) if all_recall else 0.0,
            "precision": float(sum(all_precision) / len(all_precision)) if all_precision else 0.0,
            "num_queries": len(results)
        }
        
        return results, overall_metrics
    
    def _preprocess(self, query_text: str) -> List[str]:
        """Preprocess query text"""
        try:
            response = requests.post(
                f"{self.preprocessing_service_url}/preprocess",
                json={"text": query_text, "mode": "query"},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("tokens", [])
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return query_text.lower().split()
