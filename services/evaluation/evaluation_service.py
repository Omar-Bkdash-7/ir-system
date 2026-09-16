"""
Evaluation service - main business logic.
Orchestrates evaluation of different retrieval models.
"""

import logging
from typing import Dict, List

from services.evaluation.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationMetrics,
    QueryEvaluationResult
)
from services.evaluation.services.retrieval_evaluator import RetrievalEvaluator
from services.evaluation.services.hybrid_evaluator import HybridEvaluator
from services.evaluation.services.results_storage import ResultsStorage
from services.evaluation.engines.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class EvaluationService:
    """Main evaluation service coordinating all evaluation tasks"""
    
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.hybrid_evaluator = HybridEvaluator()
        self.dataset_loader = DatasetLoader()
        self.results_storage = ResultsStorage()
    
    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """
        Evaluate a retrieval model.
        
        Args:
            request: Evaluation request
            
        Returns:
            Evaluation response with metrics
        """
        logger.info(f"Starting evaluation: model={request.model}, dataset={request.dataset_name}")
        
        if not self.dataset_loader.validate_dataset(request.dataset_name):
            raise ValueError(f"Dataset {request.dataset_name} is not valid for evaluation")
        
        if request.model in ["tfidf", "embedding", "bm25"]:
            results, metrics = self.retrieval_evaluator.evaluate_model(
                model=request.model,
                dataset_name=request.dataset_name,
                top_k=request.top_k,
                query_limit=request.query_limit,
                bm25_k1=request.bm25_k1,
                bm25_b=request.bm25_b,
                enable_refinement=request.enable_refinement
            )
        
        elif request.model == "hybrid_serial":
            if not request.hybrid_models or len(request.hybrid_models) < 2:
                raise ValueError("hybrid_serial requires at least 2 models in hybrid_models")
            
            results, metrics = self.hybrid_evaluator.evaluate_hybrid_serial(
                dataset_name=request.dataset_name,
                first_model=request.hybrid_models[0],
                second_model=request.hybrid_models[1],
                top_k=request.top_k,
                query_limit=request.query_limit,
                enable_refinement=request.enable_refinement
            )
        
        elif request.model == "hybrid_parallel":
            if not request.hybrid_models or len(request.hybrid_models) < 2:
                raise ValueError("hybrid_parallel requires at least 2 models in hybrid_models")
            
            results, metrics = self.hybrid_evaluator.evaluate_hybrid_parallel(
                dataset_name=request.dataset_name,
                models=request.hybrid_models,
                fusion_method=request.fusion_method,
                top_k=request.top_k,
                query_limit=request.query_limit,
                enable_refinement=request.enable_refinement
            )
        
        else:
            raise ValueError(f"Unknown model: {request.model}")
        
        evaluation_metrics = EvaluationMetrics(
            map=metrics["map"],
            ndcg=metrics["ndcg"],
            recall=metrics["recall"],
            precision=metrics["precision"],
            num_queries=metrics["num_queries"]
        )
        
        # Save results to database
        try:
            metadata = {
                "dataset": request.dataset_name,
                "query_limit": request.query_limit,
                "enable_refinement": request.enable_refinement
            }
            if request.model in ["hybrid_serial", "hybrid_parallel"]:
                metadata["hybrid_models"] = request.hybrid_models
                metadata["fusion_method"] = request.fusion_method
            if request.model == "bm25":
                metadata["bm25_k1"] = request.bm25_k1
                metadata["bm25_b"] = request.bm25_b
            
            self.results_storage.save_results(
                model=request.model,
                dataset=request.dataset_name,
                top_k=request.top_k,
                map_score=metrics["map"],
                ndcg_score=metrics["ndcg"],
                recall_score=metrics["recall"],
                precision_score=metrics["precision"],
                num_queries=metrics["num_queries"],
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to save results to database: {e}")
        
        query_results = None
        if request.include_per_query:
            query_results = [
                QueryEvaluationResult(
                    query_id=query_id,
                    query_text=data["query_text"],
                    num_retrieved=data["num_retrieved"],
                    num_relevant=data["num_relevant"],
                    num_relevant_retrieved=data["num_relevant_retrieved"],
                    ap=data["ap"],
                    ndcg=data["ndcg"],
                    recall=data["recall"],
                    precision=data["precision"]
                )
                for query_id, data in results.items()
            ]
        
        return EvaluationResponse(
            model=request.model,
            top_k=request.top_k,
            metrics=evaluation_metrics,
            query_results=query_results
        )
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets for evaluation"""
        return [
            "beir/quora/test",
            "beir/trec-covid",
            "beir/nfcorpus",
            "beir/scifact",
            "beir/fiqa"
        ]
    
    def get_evaluation_history(
        self,
        model: str = None,
        dataset: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get evaluation history from database"""
        return self.results_storage.get_latest_results(
            model=model,
            dataset=dataset,
            limit=limit
        )
    
    def compare_models(self, dataset: str, top_k: int = 100) -> List[Dict]:
        """Compare all models for a specific dataset"""
        return self.results_storage.compare_models(dataset=dataset, top_k=top_k)
    
    def get_latest_metrics(self) -> Dict:
        """
        Get latest evaluation metrics for all models.
        Returns in format expected by frontend: {model_name: {metric: value}}
        """
        results = self.results_storage.get_latest_results(limit=1)
        
        if not results:
            return {}
        
        metrics_by_model = {}
        for result in results:
            model_name = result["model"].upper().replace("_", "-")
            if model_name == "TFIDF":
                model_name = "TF-IDF"
            
            metrics_by_model[model_name] = {
                "map": result["map"],
                "ndcg_at_10": result["ndcg"],
                "recall": result["recall"],
                "precision_at_10": result["precision"],
                "num_queries": result["num_queries"]
            }
        
        all_history = self.results_storage.get_latest_results(limit=100)
        unique_models = set(r["model"] for r in all_history)
        
        for model in unique_models:
            if model not in [k.lower().replace("-", "_") for k in metrics_by_model.keys()]:
                model_results = [r for r in all_history if r["model"] == model]
                if model_results:
                    latest = model_results[0]
                    model_name = model.upper().replace("_", "-")
                    if model_name == "TFIDF":
                        model_name = "TF-IDF"
                    
                    metrics_by_model[model_name] = {
                        "map": latest["map"],
                        "ndcg_at_10": latest["ndcg"],
                        "recall": latest["recall"],
                        "precision_at_10": latest["precision"],
                        "num_queries": latest["num_queries"]
                    }
        
        return metrics_by_model
    
    def get_per_query_analysis(self) -> Dict:
        """Get best/worst performing queries per model"""
        return {
            "message": "Per-query analysis not yet implemented",
            "available": False
        }
    
    def get_refinement_comparison(self) -> Dict:
        """Get metrics comparison before/after query refinement"""
        return {
            "message": "Refinement comparison requires separate evaluation runs",
            "available": False,
            "before": {},
            "after": {}
        }
