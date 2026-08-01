"""
Results storage service.
Handles saving evaluation results to database.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from common.database import get_db_connection, execute_query

logger = logging.getLogger(__name__)


class ResultsStorage:
    """Store and retrieve evaluation results from database"""
    
    def __init__(self):
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure evaluation_results table exists"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id SERIAL PRIMARY KEY,
            model VARCHAR(100) NOT NULL,
            dataset VARCHAR(200) NOT NULL,
            top_k INTEGER NOT NULL,
            map_score FLOAT,
            ndcg_score FLOAT,
            recall_score FLOAT,
            precision_score FLOAT,
            num_queries INTEGER,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluation_results(model);
        CREATE INDEX IF NOT EXISTS idx_eval_dataset ON evaluation_results(dataset);
        CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluation_results(created_at DESC);
        """
        
        try:
            conn = get_db_connection()
            execute_query(conn, create_table_query)
            conn.close()
            logger.info("Evaluation results table verified")
        except Exception as e:
            logger.error(f"Failed to create evaluation_results table: {e}")
    
    def save_results(
        self,
        model: str,
        dataset: str,
        top_k: int,
        map_score: float,
        ndcg_score: float,
        recall_score: float,
        precision_score: float,
        num_queries: int,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Save evaluation results to database.
        
        Args:
            model: Model name
            dataset: Dataset name
            top_k: Top K parameter
            map_score: MAP score
            ndcg_score: nDCG score
            recall_score: Recall score
            precision_score: Precision score
            num_queries: Number of queries evaluated
            metadata: Additional metadata (optional)
            
        Returns:
            ID of saved record
        """
        import json
        
        query = """
        INSERT INTO evaluation_results 
        (model, dataset, top_k, map_score, ndcg_score, recall_score, 
         precision_score, num_queries, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, (
                model, dataset, top_k, map_score, ndcg_score,
                recall_score, precision_score, num_queries, metadata_json
            ))
            result_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Saved evaluation results with ID: {result_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
            raise
    
    def get_latest_results(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: int = 10
    ) -> list:
        """
        Get latest evaluation results.
        
        Args:
            model: Filter by model (optional)
            dataset: Filter by dataset (optional)
            limit: Maximum number of results
            
        Returns:
            List of evaluation results
        """
        query = """
        SELECT id, model, dataset, top_k, map_score, ndcg_score,
               recall_score, precision_score, num_queries, created_at
        FROM evaluation_results
        WHERE 1=1
        """
        params = []
        
        if model:
            query += " AND model = %s"
            params.append(model)
        
        if dataset:
            query += " AND dataset = %s"
            params.append(dataset)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "model": row[1],
                    "dataset": row[2],
                    "top_k": row[3],
                    "map": row[4],
                    "ndcg": row[5],
                    "recall": row[6],
                    "precision": row[7],
                    "num_queries": row[8],
                    "created_at": row[9].isoformat() if row[9] else None
                })
            
            cursor.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get evaluation results: {e}")
            return []
    
    def compare_models(
        self,
        dataset: str,
        top_k: int = 100
    ) -> list:
        """
        Compare all models for a specific dataset.
        
        Args:
            dataset: Dataset name
            top_k: Top K parameter
            
        Returns:
            List of model comparisons
        """
        query = """
        SELECT model, 
               AVG(map_score) as avg_map,
               AVG(ndcg_score) as avg_ndcg,
               AVG(recall_score) as avg_recall,
               AVG(precision_score) as avg_precision,
               COUNT(*) as num_evaluations
        FROM evaluation_results
        WHERE dataset = %s AND top_k = %s
        GROUP BY model
        ORDER BY avg_map DESC;
        """
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, (dataset, top_k))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "model": row[0],
                    "avg_map": float(row[1]) if row[1] else 0.0,
                    "avg_ndcg": float(row[2]) if row[2] else 0.0,
                    "avg_recall": float(row[3]) if row[3] else 0.0,
                    "avg_precision": float(row[4]) if row[4] else 0.0,
                    "num_evaluations": row[5]
                })
            
            cursor.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return []
