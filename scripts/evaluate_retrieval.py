import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ir_datasets
import json
from services.retrieval.tfidf_retriever import TFIDFRetriever
from services.retrieval.embedding_retriever import EmbeddingRetriever
from services.preprocessing.preprocessing_service import preprocess
from common.logging_config import setup_logging
from tqdm import tqdm
import numpy as np

logger = setup_logging("evaluation")


def load_dataset_queries_and_qrels(dataset_name: str = "beir/quora/test"):
    """Load queries and qrels from ir_datasets"""
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = ir_datasets.load(dataset_name)
    
    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
    
    qrels = {}
    if hasattr(dataset, 'qrels_iter'):
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        logger.info(f"Loaded {len(queries)} queries and {len(qrels)} qrels")
    else:
        logger.warning(f"Dataset {dataset_name} does not have qrels. Evaluation will be limited.")
        logger.info(f"Loaded {len(queries)} queries only")
    
    return queries, qrels


def evaluate_tfidf(queries: dict, qrels: dict, top_k: int = 100):
    """Evaluate TF-IDF retrieval"""
    logger.info("Starting TF-IDF evaluation")
    
    retriever = TFIDFRetriever()
    retriever.load()
    
    results = {}
    
    for query_id, query_text in tqdm(queries.items(), desc="TF-IDF Retrieval"):
        # Preprocess query
        tokens = preprocess(query_text, mode="query")
        
        if not tokens:
            logger.warning(f"Empty tokens for query {query_id}: {query_text}")
            results[query_id] = {
                "query_text": query_text,
                "retrieved": []
            }
            continue
        
        # Retrieve
        retrieved = retriever.retrieve(tokens, top_k=top_k)
        
        # Add document texts from database
        if retrieved:
            try:
                from common.database import get_documents_by_ids
                doc_ids = [doc["doc_id"] for doc in retrieved]
                doc_texts = get_documents_by_ids(doc_ids)
                
                for rank, doc in enumerate(retrieved, 1):
                    doc["rank"] = rank
                    doc["text"] = doc_texts.get(doc["doc_id"], "")
            except Exception as e:
                logger.error(f"Failed to fetch document texts from database: {e}")
                for rank, doc in enumerate(retrieved, 1):
                    doc["rank"] = rank
                    doc["text"] = "Text unavailable (DB connection error)"
        
        results[query_id] = {
            "query_text": query_text,
            "retrieved": retrieved
        }
    
    # Calculate metrics
    metrics = calculate_metrics(results, qrels, top_k)
    
    logger.info("TF-IDF Evaluation complete")
    return results, metrics


def evaluate_embedding(queries: dict, qrels: dict, top_k: int = 100):
    """Evaluate Embedding retrieval"""
    logger.info("Starting Embedding evaluation")
    
    retriever = EmbeddingRetriever()
    retriever.load()
    
    results = {}
    
    for query_id, query_text in tqdm(queries.items(), desc="Embedding Retrieval"):
        # Retrieve (preprocessing is done inside embedding_retriever now)
        retrieved = retriever.retrieve(query_text, top_k=top_k)
        
        # Add document texts from database
        if retrieved:
            try:
                from common.database import get_documents_by_ids
                doc_ids = [doc["doc_id"] for doc in retrieved]
                doc_texts = get_documents_by_ids(doc_ids)
                
                for rank, doc in enumerate(retrieved, 1):
                    doc["rank"] = rank
                    doc["text"] = doc_texts.get(doc["doc_id"], "")
            except Exception as e:
                logger.error(f"Failed to fetch document texts from database: {e}")
                for rank, doc in enumerate(retrieved, 1):
                    doc["rank"] = rank
                    doc["text"] = "Text unavailable (DB connection error)"
        
        results[query_id] = {
            "query_text": query_text,
            "retrieved": retrieved
        }
    
    # Calculate metrics
    metrics = calculate_metrics(results, qrels, top_k)
    
    logger.info("Embedding Evaluation complete")
    return results, metrics


def calculate_metrics(results: dict, qrels: dict, top_k: int = 100):
    """Calculate IR metrics: nDCG@k, MAP, Recall@k, Precision@k"""
    
    ndcg_scores = []
    map_scores = []
    recall_scores = []
    precision_scores = []
    
    for query_id, query_data in results.items():
        if query_id not in qrels or not qrels[query_id]:
            continue
        
        relevant_docs = qrels[query_id]
        retrieved_list = query_data.get("retrieved", [])[:top_k]
        
        # Extractions safely from the inner list
        retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_list]
        retrieved_scores = [doc["score"] for doc in retrieved_list]
        
        # Calculate nDCG@k
        ndcg = calculate_ndcg(retrieved_doc_ids, retrieved_scores, relevant_docs, top_k)
        ndcg_scores.append(ndcg)
        
        # Calculate MAP
        ap = calculate_average_precision(retrieved_doc_ids, relevant_docs)
        map_scores.append(ap)
        
        # Calculate Recall@k
        recall = calculate_recall(retrieved_doc_ids, relevant_docs)
        recall_scores.append(recall)
        
        # Calculate Precision@k
        precision = calculate_precision(retrieved_doc_ids, relevant_docs)
        precision_scores.append(precision)
    
    metrics = {
        f"nDCG@{top_k}": np.mean(ndcg_scores) if ndcg_scores else 0.0,
        "MAP": np.mean(map_scores) if map_scores else 0.0,
        f"Recall@{top_k}": np.mean(recall_scores) if recall_scores else 0.0,
        f"Precision@{top_k}": np.mean(precision_scores) if precision_scores else 0.0,
        "num_queries": len(ndcg_scores)
    }
    
    return metrics


def calculate_ndcg(retrieved_doc_ids: list, retrieved_scores: list, relevant_docs: dict, k: int):
    """Calculate Normalized Discounted Cumulative Gain"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in relevant_docs:
            rel = relevant_docs[doc_id]
            dcg += (2**rel - 1) / np.log2(i + 2)
    
    # Ideal DCG
    ideal_rels = sorted(relevant_docs.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_average_precision(retrieved_doc_ids: list, relevant_docs: dict):
    """Calculate Average Precision"""
    num_relevant = 0
    sum_precision = 0.0
    
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            sum_precision += precision_at_i
    
    total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
    return sum_precision / total_relevant if total_relevant > 0 else 0.0


def calculate_recall(retrieved_doc_ids: list, relevant_docs: dict):
    """Calculate Recall@k"""
    relevant_retrieved = sum(1 for doc_id in retrieved_doc_ids 
                            if doc_id in relevant_docs and relevant_docs[doc_id] > 0)
    total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
    
    return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0


def calculate_precision(retrieved_doc_ids: list, relevant_docs: dict):
    """Calculate Precision@k"""
    relevant_retrieved = sum(1 for doc_id in retrieved_doc_ids 
                            if doc_id in relevant_docs and relevant_docs[doc_id] > 0)
    
    return relevant_retrieved / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0


def save_results(model_name: str, results: dict, metrics: dict, output_dir: str = "data/evaluation"):
    """Save evaluation results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_path / f"{model_name}_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save metrics
    metrics_file = output_path / f"{model_name}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    # Print metrics
    print(f"\n{'='*50}")
    print(f"{model_name.upper()} Evaluation Results")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"{'='*50}\n")


def main():
    # Load dataset
    queries, qrels = load_dataset_queries_and_qrels("beir/quora/test")
    
    # Limit to first 10 queries for faster evaluation
    queries = dict(list(queries.items())[:10])
    
    # Evaluate TF-IDF
    logger.info("="*50)
    logger.info("Evaluating TF-IDF")
    logger.info("="*50)
    tfidf_results, tfidf_metrics = evaluate_tfidf(queries, qrels, top_k=100)
    save_results("tfidf", tfidf_results, tfidf_metrics)
    
    # Evaluate Embedding
    logger.info("="*50)
    logger.info("Evaluating Embedding")
    logger.info("="*50)
    embedding_results, embedding_metrics = evaluate_embedding(queries, qrels, top_k=100)
    save_results("embedding", embedding_results, embedding_metrics)
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Metric':<20} {'TF-IDF':<15} {'Embedding':<15}")
    print("-"*50)
    for metric in tfidf_metrics.keys():
        if metric != "num_queries":
            tfidf_val = tfidf_metrics[metric]
            emb_val = embedding_metrics[metric]
            print(f"{metric:<20} {tfidf_val:<15.4f} {emb_val:<15.4f}")
    print("="*50)
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()