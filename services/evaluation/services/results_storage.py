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
        Replaces existing result if same model, dataset, and refinement setting exists.
        
        Args:
            model: Model name
            dataset: Dataset name
            top_k: Top K parameter
            map_score: MAP score
            ndcg_score: nDCG score
            recall_score: Recall score
            precision_score: Precision score
            num_queries: Number of queries evaluated
            metadata: Additional metadata (must include enable_refinement)
            
        Returns:
            ID of saved record
        """
        import json
        
        # Extract refinement setting from metadata
        enable_refinement = False
        if metadata and isinstance(metadata, dict):
            enable_refinement = metadata.get("enable_refinement", False)
        
        # Check if record exists with same model, dataset, and refinement setting
        check_query = """
        SELECT id FROM evaluation_results 
        WHERE model = %s AND dataset = %s 
        AND (metadata->>'enable_refinement')::boolean = %s
        LIMIT 1;
        """
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check for existing record
            cursor.execute(check_query, (model, dataset, enable_refinement))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                update_query = """
                UPDATE evaluation_results 
                SET top_k = %s, map_score = %s, ndcg_score = %s, 
                    recall_score = %s, precision_score = %s, num_queries = %s,
                    metadata = %s, created_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING id;
                """
                cursor.execute(update_query, (
                    top_k, map_score, ndcg_score, recall_score, 
                    precision_score, num_queries, metadata_json, existing[0]
                ))
                result_id = cursor.fetchone()[0]
                logger.info(f"Updated existing evaluation result with ID: {result_id}")
            else:
                # Insert new record
                insert_query = """
                INSERT INTO evaluation_results 
                (model, dataset, top_k, map_score, ndcg_score, recall_score, 
                 precision_score, num_queries, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                cursor.execute(insert_query, (
                    model, dataset, top_k, map_score, ndcg_score,
                    recall_score, precision_score, num_queries, metadata_json
                ))
                result_id = cursor.fetchone()[0]
                logger.info(f"Saved new evaluation result with ID: {result_id}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
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
               recall_score, precision_score, num_queries, metadata, created_at
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
                    "metadata": row[9],
                    "created_at": row[10].isoformat() if row[10] else None
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
    
    def delete_result(self, result_id: int) -> bool:
        """
        Delete a specific evaluation result by ID.
        
        Args:
            result_id: ID of the result to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        query = "DELETE FROM evaluation_results WHERE id = %s RETURNING id;"
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, (result_id,))
            deleted = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()
            
            if deleted:
                logger.info(f"Deleted evaluation result with ID: {result_id}")
                return True
            else:
                logger.warning(f"Evaluation result with ID {result_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete evaluation result {result_id}: {e}")
            return False
    
    def delete_by_filters(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        before_date: Optional[str] = None
    ) -> int:
        """
        Delete evaluation results by filters.
        
        Args:
            model: Filter by model name
            dataset: Filter by dataset name
            before_date: Delete results before this date (ISO format)
            
        Returns:
            Number of deleted records
        """
        query = "DELETE FROM evaluation_results WHERE 1=1"
        params = []
        
        if model:
            query += " AND model = %s"
            params.append(model)
        
        if dataset:
            query += " AND dataset = %s"
            params.append(dataset)
        
        if before_date:
            query += " AND created_at < %s"
            params.append(before_date)
        
        query += " RETURNING id;"
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            deleted_ids = cursor.fetchall()
            count = len(deleted_ids)
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Deleted {count} evaluation results with filters: model={model}, dataset={dataset}, before={before_date}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete evaluation results by filters: {e}")
            return 0
    
    def clear_all(self) -> int:
        """
        Delete all evaluation results.
        WARNING: This cannot be undone!
        
        Returns:
            Number of deleted records
        """
        query = "DELETE FROM evaluation_results RETURNING id;"
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            deleted_ids = cursor.fetchall()
            count = len(deleted_ids)
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.warning(f"Cleared all evaluation results: {count} records deleted")
            return count
            
        except Exception as e:
            logger.error(f"Failed to clear all evaluation results: {e}")
            return 0
