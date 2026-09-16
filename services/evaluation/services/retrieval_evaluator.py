"""
Retrieval evaluator service.
Handles evaluation of different retrieval models.
"""

import requests
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from common.config import get_config
from services.evaluation.engines.metrics_calculator import MetricsCalculator
from services.evaluation.engines.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluate retrieval models using standard IR metrics"""
    
    def __init__(self):
        self.retrieval_service_url = get_config(
            "services.retrieval_url",
            "http://retrieval:8003"
        )
        self.preprocessing_service_url = get_config(
            "services.preprocessing_url",
            "http://preprocessing:8001"
        )
        self.query_refinement_service_url = get_config(
            "services.query_refinement_url",
            "http://query_refinement:8005"
        )
        self.metrics_calculator = MetricsCalculator()
        self.dataset_loader = DatasetLoader()
    
    def evaluate_model(
        self,
        model: str,
        dataset_name: str,
        top_k: int = 100,
        query_limit: Optional[int] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        enable_refinement: bool = False
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate a single retrieval model.
        
        Args:
            model: Model name ('tfidf', 'embedding', 'bm25')
            dataset_name: Dataset identifier
            top_k: Number of documents to retrieve
            query_limit: Limit number of queries (for testing)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            enable_refinement: Apply query refinement
            
        Returns:
            Tuple of (results, metrics)
        """
        logger.info(f"Evaluating {model} model on {dataset_name} (refinement={'enabled' if enable_refinement else 'disabled'})")
        
        queries, qrels = self.dataset_loader.load_queries_and_qrels(dataset_name)
        
        if query_limit:
            queries = dict(list(queries.items())[:query_limit])
            logger.info(f"Limited to {query_limit} queries")
        
        results = {}
        all_ndcg = []
        all_ap = []
        all_recall = []
        all_precision = []
        
        for query_id, query_text in tqdm(queries.items(), desc=f"Evaluating {model}"):
            if query_id not in qrels or not qrels[query_id]:
                logger.debug(f"Query {query_id} has no qrels, skipping")
                continue
            
            try:
                # Apply query refinement if enabled
                refined_query = query_text
                if enable_refinement:
                    refined_query = self._refine_query(query_text)
                
                retrieved_docs = self._retrieve(
                    query_text=refined_query,
                    model=model,
                    top_k=top_k,
                    bm25_k1=bm25_k1,
                    bm25_b=bm25_b,
                    dataset_name=dataset_name
                )
                
                if not retrieved_docs:
                    logger.warning(f"No documents retrieved for query {query_id}")
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
        
        logger.info(f"Evaluation complete: {overall_metrics}")
        return results, overall_metrics
    
    def _retrieve(
        self,
        query_text: str,
        model: str,
        top_k: int,
        bm25_k1: float,
        bm25_b: float,
        dataset_name: str
    ) -> List[Dict]:
        """
        Retrieve documents using the specified model.
        
        Args:
            query_text: Query text
            model: Model name
            top_k: Number of documents to retrieve
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            dataset_name: Dataset name
            
        Returns:
            List of retrieved documents
        """
        if model == "embedding":
            payload = {
                "query_text": query_text,
                "top_k": top_k,
                "include_text": False,
                "dataset": dataset_name
            }
            endpoint = f"{self.retrieval_service_url}/retrieve/embedding"
        
        elif model == "bm25":
            query_tokens = self._preprocess(query_text)
            payload = {
                "query_tokens": query_tokens,
                "query_text": query_text,
                "top_k": top_k,
                "include_text": False,
                "bm25_k1": bm25_k1,
                "bm25_b": bm25_b,
                "dataset": dataset_name
            }
            endpoint = f"{self.retrieval_service_url}/retrieve/bm25"
        
        else:
            query_tokens = self._preprocess(query_text)
            payload = {
                "query_tokens": query_tokens,
                "top_k": top_k,
                "include_text": False,
                "dataset": dataset_name
            }
            endpoint = f"{self.retrieval_service_url}/retrieve/tfidf"
        
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data.get("results", [])
    
    def _preprocess(self, query_text: str) -> List[str]:
        """
        Preprocess query text using preprocessing service.
        
        Args:
            query_text: Query text
            
        Returns:
            List of tokens
        """
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
    
    def _refine_query(self, query_text: str) -> str:
        """
        Refine query text using query refinement service.
        
        Args:
            query_text: Original query text
            
        Returns:
            Refined query text
        """
        try:
            response = requests.post(
                f"{self.query_refinement_service_url}/refine",
                json={
                    "query": query_text,
                    "history": [],
                    "enabled": True
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            refined = result.get("refined_query", query_text)
            
            if refined != query_text:
                logger.debug(f"Query refined: '{query_text}' -> '{refined}'")
            
            return refined
        except Exception as e:
            logger.warning(f"Query refinement failed, using original query: {e}")
            return query_text
