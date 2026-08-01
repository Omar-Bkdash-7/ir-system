from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import scipy.sparse
import jsonlines
import json
from pathlib import Path
from common.config import get_config
import logging

logger = logging.getLogger(__name__)

class TFIDFIndexer:
    def __init__(self, dataset_name: str = None):
        self.vectorizer = None
        self.matrix = None
        self.doc_ids = []
        
        # Use dataset-specific paths
        if dataset_name:
            dataset_slug = dataset_name.replace('/', '_').replace('-', '_')
            base_path = Path(get_config("indices.base_path", "data/indices")) / dataset_slug / "tfidf"
        else:
            # Fallback to old behavior for backwards compatibility
            base_path = Path(get_config("indices.base_path", "data/indices"))
        
        self.vectorizer_path = base_path / "tfidf_vectorizer.pkl"
        self.matrix_path = base_path / "tfidf_matrix.npz"
        self.doc_ids_path = base_path / "tfidf_doc_ids.json"
    
    def build(self, processed_path: str = None):
        if processed_path is None:
            processed_path = get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl")
        
        processed_path = Path(processed_path)
        if not processed_path.exists():
            raise FileNotFoundError(f"Preprocessed documents not found at {processed_path}")
        
        logger.info(f"Loading preprocessed documents from {processed_path}")
        
        doc_ids = []
        corpus = []
        
        with jsonlines.open(processed_path) as reader:
            for doc in reader:
                doc_ids.append(doc["doc_id"])
                corpus.append(" ".join(doc["tokens"]))
        
        logger.info(f"Building TF-IDF index for {len(doc_ids)} documents")
        
        sublinear_tf = get_config("models.tfidf.sublinear_tf", True)
        norm = get_config("models.tfidf.norm", "l2")
        
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=sublinear_tf,
            norm=norm
        )
        
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.doc_ids = doc_ids
        
        logger.info(f"TF-IDF matrix shape: {self.matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return self
    
    def save(self):
        self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Saved vectorizer to {self.vectorizer_path}")
        
        scipy.sparse.save_npz(self.matrix_path, self.matrix)
        logger.info(f"Saved TF-IDF matrix to {self.matrix_path}")
        
        with open(self.doc_ids_path, "w") as f:
            json.dump(self.doc_ids, f)
        logger.info(f"Saved doc IDs to {self.doc_ids_path}")
    
    def load(self):
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {self.vectorizer_path}")
        
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        
        self.matrix = scipy.sparse.load_npz(self.matrix_path)
        
        with open(self.doc_ids_path, "r") as f:
            self.doc_ids = json.load(f)
        
        logger.info(f"Loaded TF-IDF index: {self.matrix.shape[0]} docs, {self.matrix.shape[1]} features")
        
        return self
    
    def exists(self) -> bool:
        return (self.vectorizer_path.exists() and 
                self.matrix_path.exists() and 
                self.doc_ids_path.exists())
