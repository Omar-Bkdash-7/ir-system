import pickle
import scipy.sparse
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from common.config import get_active_dataset_path, get_config
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
        
        try:
            vectorizer_path = Path(get_active_dataset_path("indices.tfidf_vectorizer", "data/indices/tfidf_vectorizer.pkl"))
            matrix_path = Path(get_active_dataset_path("indices.tfidf_matrix", "data/indices/tfidf_matrix.npz"))
            doc_ids_path = Path(get_active_dataset_path("indices.tfidf_doc_ids", "data/indices/tfidf_doc_ids.json"))
        except ValueError as e:
            raise FileNotFoundError(f"Cannot load TF-IDF index: {e}")
        
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
        
        # Apply preprocessing to query tokens (they are already tokens, so join and preprocess)
        from services.preprocessing.preprocessing_service import preprocess
        query_text = " ".join(query_tokens)
        processed_tokens = preprocess(query_text, mode="query")
        
        if processed_tokens:
            query_str = " ".join(processed_tokens)
            logger.debug(f"Original tokens: {query_tokens[:5]}...")
            logger.debug(f"Preprocessed tokens: {processed_tokens[:5]}...")
        else:
            query_str = query_text
            logger.warning("Query preprocessing resulted in empty tokens, using original")
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
    
    def score_documents(self, query_tokens: list[str], doc_ids: list[str]) -> dict[str, float]:
        if not self.loaded:
            self.load()
        
        if not query_tokens or not doc_ids:
            return {}
        
        from services.preprocessing.preprocessing_service import preprocess
        query_text = " ".join(query_tokens)
        processed_tokens = preprocess(query_text, mode="query")
        
        if processed_tokens:
            query_str = " ".join(processed_tokens)
        else:
            query_str = query_text
        
        q_vec = self.vectorizer.transform([query_str])
        
        if q_vec.nnz == 0:
            logger.warning("Query produces all-zero vector for scoring")
            return {doc_id: 0.0 for doc_id in doc_ids}
        
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        
        doc_indices = []
        valid_doc_ids = []
        for doc_id in doc_ids:
            if doc_id in doc_id_to_idx:
                doc_indices.append(doc_id_to_idx[doc_id])
                valid_doc_ids.append(doc_id)
        
        if not doc_indices:
            logger.warning("None of the provided doc_ids found in index")
            return {doc_id: 0.0 for doc_id in doc_ids}
        
        doc_vecs = self.matrix[doc_indices]
        
        scores = cosine_similarity(q_vec, doc_vecs).flatten()
        
        result = {valid_doc_ids[i]: float(scores[i]) for i in range(len(valid_doc_ids))}
        
        for doc_id in doc_ids:
            if doc_id not in result:
                result[doc_id] = 0.0
        
        logger.debug(f"Scored {len(result)} documents")
        
        return result
