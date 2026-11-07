# Snapshot of the retriever as it was BEFORE the rewrite, kept only as the reference for tests/retrieval/test_differential.py.
# Do not import it from the service.
import pickle
import heapq
from pathlib import Path
from typing import List, Dict, NamedTuple
from common.config import get_dataset_specific_path, get_config
from services.preprocessing.preprocessing_service import preprocess
import logging
import math

logger = logging.getLogger(__name__)

class PostingEntry(NamedTuple):
    doc_id: str
    term_freq: int

class BM25Retriever:
    def __init__(self, dataset_name: str):
        """
        Initialize BM25 retriever.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        self.dataset_name = dataset_name
        self.inverted_index: Dict[str, List[PostingEntry]] = {}
        self.doc_frequencies: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.doc_ids: List[str] = []
        self.loaded = False
    
    def load(self):
        if self.loaded:
            return
        
        try:
            index_path = Path(get_dataset_specific_path("indices.bm25_index", self.dataset_name, "data/indices/bm25_index.pkl"))
        except ValueError as e:
            raise FileNotFoundError(f"Cannot load BM25 index: {e}")
        
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {index_path}")
        
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)
        
        self.inverted_index = {term: [PostingEntry(doc_id, tf) for doc_id, tf in entries]
                               for term, entries in index_data["inverted_index"].items()}
        self.doc_frequencies = index_data["doc_frequencies"]
        self.idf_scores = index_data.get("idf_scores", {})
        self.doc_lengths = index_data["doc_lengths"]
        self.avg_doc_length = index_data["avg_doc_length"]
        self.total_docs = index_data["total_docs"]
        self.doc_ids = index_data["doc_ids"]
        
        if not self.idf_scores:
            logger.warning("IDF scores not in index, computing on-the-fly")
        
        self.loaded = True
        logger.info(f"BM25 retriever loaded: {self.total_docs} documents, {len(self.inverted_index)} terms, avg_len={self.avg_doc_length:.2f}")
    
    def compute_idf(self, term: str) -> float:
        if term in self.idf_scores:
            return self.idf_scores[term]
        
        if term not in self.doc_frequencies:
            return 0.0
        
        df = self.doc_frequencies[term]
        idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
        return idf
    
    def retrieve(self, query_tokens: List[str], top_k: int = None, k1: float = 1.5, b: float = 0.75) -> List[Dict]:
        if not self.loaded:
            self.load()
        
        if top_k is None:
            top_k = get_config("retrieval.candidates_k", 1000)
        
        if not query_tokens:
            logger.warning("Empty query tokens provided")
            return []
        
        query_text = " ".join(query_tokens)
        processed_tokens = preprocess(query_text, mode="query")
        
        if not processed_tokens:
            logger.warning("Query preprocessing resulted in empty tokens")
            return []
        
        logger.debug(f"BM25 query tokens: {processed_tokens[:10]}")
        
        if self.avg_doc_length <= 0:
            logger.error("Invalid avg_doc_length, using fallback value")
            self.avg_doc_length = 1.0
        
        doc_scores: Dict[str, float] = {}
        
        for term in processed_tokens:
            if term not in self.inverted_index:
                logger.debug(f"Query term '{term}' not in index (IDF=0)")
                continue
            
            idf = self.compute_idf(term)
            posting_list = self.inverted_index[term]
            
            for entry in posting_list:
                doc_id = entry.doc_id
                tf = entry.term_freq
                doc_len = self.doc_lengths.get(doc_id, self.avg_doc_length)
                
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_length))
                
                score_contribution = idf * (numerator / denominator)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += score_contribution
        
        if not doc_scores:
            logger.warning("No documents matched the query")
            return []
        
        top_docs = heapq.nlargest(top_k, doc_scores.items(), key=lambda x: x[1])
        
        results = [
            {
                "doc_id": doc_id,
                "score": float(score)
            }
            for doc_id, score in top_docs
        ]
        
        logger.info(f"Retrieved {len(results)} documents for query with BM25 (k1={k1}, b={b})")
        
        return results
    
    def get_statistics(self) -> dict:
        if not self.loaded:
            self.load()
        
        return {
            "total_docs": self.total_docs,
            "vocab_size": len(self.inverted_index),
            "avg_doc_length": round(self.avg_doc_length, 2)
        }
    
    def score_documents(self, query_tokens: List[str], doc_ids: List[str], k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        if not self.loaded:
            self.load()
        
        if not query_tokens or not doc_ids:
            return {}
        
        query_text = " ".join(query_tokens)
        processed_tokens = preprocess(query_text, mode="query")
        
        if not processed_tokens:
            logger.warning("Query preprocessing resulted in empty tokens")
            return {doc_id: 0.0 for doc_id in doc_ids}
        
        if self.avg_doc_length <= 0:
            self.avg_doc_length = 1.0
        
        doc_id_set = set(doc_ids)
        doc_scores: Dict[str, float] = {doc_id: 0.0 for doc_id in doc_ids}
        
        for term in processed_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self.compute_idf(term)
            posting_list = self.inverted_index[term]
            
            for entry in posting_list:
                doc_id = entry.doc_id
                
                if doc_id not in doc_id_set:
                    continue
                
                tf = entry.term_freq
                doc_len = self.doc_lengths.get(doc_id, self.avg_doc_length)
                
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_length))
                
                score_contribution = idf * (numerator / denominator)
                doc_scores[doc_id] += score_contribution
        
        logger.debug(f"Scored {len(doc_scores)} documents with BM25")
        
        return doc_scores
