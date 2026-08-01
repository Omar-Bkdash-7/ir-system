import pickle
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, NamedTuple
import jsonlines
from common.config import get_config
import logging

logger = logging.getLogger(__name__)

class PostingEntry(NamedTuple):
    doc_id: str
    term_freq: int

class BM25Indexer:
    def __init__(self, dataset_name: str = None):
        self.inverted_index: Dict[str, List[PostingEntry]] = {}
        self.doc_frequencies: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.doc_ids: List[str] = []
        
        # Use dataset-specific paths
        if dataset_name:
            dataset_slug = dataset_name.replace('/', '_').replace('-', '_')
            base_path = Path(get_config("indices.base_path", "data/indices")) / dataset_slug / "bm25"
        else:
            # Fallback to old behavior
            base_path = Path(get_config("indices.base_path", "data/indices"))
        
        self.index_path = base_path / "bm25_index.pkl"
    
    def build(self, processed_path: str = None):
        if processed_path is None:
            processed_path = get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl")
        
        processed_path = Path(processed_path)
        if not processed_path.exists():
            raise FileNotFoundError(f"Preprocessed documents not found at {processed_path}")
        
        logger.info(f"Loading preprocessed documents from {processed_path}")
        
        temp_inverted_index = defaultdict(lambda: defaultdict(int))
        doc_ids = []
        doc_lengths = {}
        
        with jsonlines.open(processed_path) as reader:
            for doc in reader:
                doc_id = doc["doc_id"]
                tokens = doc["tokens"]
                
                doc_ids.append(doc_id)
                doc_lengths[doc_id] = len(tokens)
                
                term_counts = defaultdict(int)
                for term in tokens:
                    term_counts[term] += 1
                
                for term, tf in term_counts.items():
                    temp_inverted_index[term][doc_id] = tf
        
        self.total_docs = len(doc_ids)
        self.doc_ids = doc_ids
        self.doc_lengths = doc_lengths
        self.avg_doc_length = sum(doc_lengths.values()) / self.total_docs if self.total_docs > 0 else 0
        
        logger.info(f"Building BM25 inverted index for {self.total_docs} documents")
        logger.info(f"Average document length: {self.avg_doc_length:.2f}")
        
        for term, doc_tf_dict in temp_inverted_index.items():
            self.inverted_index[term] = [PostingEntry(doc_id, tf) for doc_id, tf in doc_tf_dict.items()]
            self.doc_frequencies[term] = len(doc_tf_dict)
        
        self._compute_idf_scores()
        
        vocab_size = len(self.inverted_index)
        logger.info(f"BM25 index built: {vocab_size} unique terms")
        
        return self
    
    def _compute_idf_scores(self):
        logger.info("Computing IDF scores for all terms...")
        for term, df in self.doc_frequencies.items():
            self.idf_scores[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
        logger.info(f"Computed IDF scores for {len(self.idf_scores)} terms")
    
    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            "inverted_index": {term: [(entry.doc_id, entry.term_freq) for entry in entries] 
                              for term, entries in self.inverted_index.items()},
            "doc_frequencies": self.doc_frequencies,
            "idf_scores": self.idf_scores,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "total_docs": self.total_docs,
            "doc_ids": self.doc_ids
        }
        
        with open(self.index_path, "wb") as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved BM25 index to {self.index_path}")
        logger.info(f"Index size: {self.index_path.stat().st_size / (1024**2):.2f} MB")
    
    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {self.index_path}")
        
        with open(self.index_path, "rb") as f:
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
            logger.warning("IDF scores not found in index, computing them now...")
            self._compute_idf_scores()
        
        logger.info(f"Loaded BM25 index: {self.total_docs} docs, {len(self.inverted_index)} terms, avg_len={self.avg_doc_length:.2f}")
        
        return self
    
    def exists(self) -> bool:
        return self.index_path.exists()
    
    def get_posting_list(self, term: str) -> List[PostingEntry]:
        return self.inverted_index.get(term, [])
    
    def get_idf(self, term: str) -> float:
        return self.idf_scores.get(term, 0.0)
    
    def compute_idf(self, term: str) -> float:
        if term in self.idf_scores:
            return self.idf_scores[term]
        
        if term not in self.doc_frequencies:
            return 0.0
        
        df = self.doc_frequencies[term]
        idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
        return idf
    
    def get_stats(self) -> dict:
        return {
            "vocab_size": len(self.inverted_index),
            "total_docs": self.total_docs,
            "avg_doc_length": round(self.avg_doc_length, 2)
        }
