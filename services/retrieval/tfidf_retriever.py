import pickle
import scipy.sparse
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from common.config import get_config
import logging

logger = logging.getLogger(__name__)

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.doc_ids = []
        self.loaded = False
    
    def load(self):
        if self.loaded:
            return
        
        vectorizer_path = Path(get_config("indices.tfidf_vectorizer", "data/indices/tfidf_vectorizer.pkl"))
        matrix_path = Path(get_config("indices.tfidf_matrix", "data/indices/tfidf_matrix.npz"))
        doc_ids_path = Path(get_config("indices.tfidf_doc_ids", "data/indices/tfidf_doc_ids.json"))
        
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}")
        
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        
        self.matrix = scipy.sparse.load_npz(matrix_path)
        
        with open(doc_ids_path, "r") as f:
            self.doc_ids = json.load(f)
        
        self.loaded = True
        logger.info(f"TF-IDF retriever loaded: {len(self.doc_ids)} documents")
    
    def retrieve(self, query_tokens: list[str], top_k: int = None) -> list[dict]:
        if not self.loaded:
            self.load()
        
        if top_k is None:
            top_k = get_config("retrieval.candidates_k", 1000)
        
        if not query_tokens:
            logger.warning("Empty query tokens provided")
            return []
        
        query_str = " ".join(query_tokens)
        q_vec = self.vectorizer.transform([query_str])
        
        if q_vec.nnz == 0:
            logger.warning(f"Query produces all-zero vector (all OOV terms): {query_str}")
            return []
        
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        
        top_idx = scores.argsort()[::-1][:top_k]
        
        results = [
            {
                "doc_id": self.doc_ids[i],
                "score": float(scores[i])
            }
            for i in top_idx if scores[i] > 0
        ]
        
        logger.info(f"Retrieved {len(results)} documents for query")
        
        return results
