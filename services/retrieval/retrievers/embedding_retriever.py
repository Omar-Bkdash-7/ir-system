import numpy as np
import json
from pathlib import Path
from common.config import get_dataset_specific_path, get_config
from common.device import get_torch_device
from common.model_loader import load_embedding_model
from common.exceptions import IndexNotFoundError, InvalidQueryError, ModelLoadError
import logging

logger = logging.getLogger(__name__)


class EmbeddingRetriever:
    def __init__(self, dataset_name: str):
        """
        Initialize Embedding retriever.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        self.dataset_name = dataset_name
        self.model = None
        self.embeddings = None
        self.doc_ids = []
        self.loaded = False
        self.device = get_torch_device()

    def load(self):
        if self.loaded:
            return

        try:
            embeddings_path = Path(get_dataset_specific_path("indices.embeddings", self.dataset_name, "data/indices/embeddings.npy"))
            doc_id_map_path = Path(get_dataset_specific_path("indices.doc_id_map", self.dataset_name, "data/indices/doc_id_map.json"))
        except ValueError as e:
            raise IndexNotFoundError(f"Cannot load embedding index: {e}")

        if not embeddings_path.exists():
            raise IndexNotFoundError(f"Embeddings not found at {embeddings_path}")
        if not doc_id_map_path.exists():
            raise IndexNotFoundError(f"Doc ID map not found at {doc_id_map_path}")

        logger.info(f"Loading embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path)

        logger.info(f"Loading doc ID map from {doc_id_map_path}")
        with open(doc_id_map_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        if len(self.doc_ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.doc_ids)} doc_ids but {self.embeddings.shape[0]} embeddings"
            )

        logger.info(f"Loading embedding model using centralized loader")

        try:
            self.model = load_embedding_model(device=self.device, cache=True)
        except ModelLoadError:
            raise

        expected_dim = self.embeddings.shape[1]
        actual_dim = self.model.get_sentence_embedding_dimension()

        if actual_dim != expected_dim:
            raise ValueError(
                f"Model dimension mismatch: model={actual_dim}, index={expected_dim}. "
                f"Rebuild the index with the current model."
            )

        self.loaded = True
        logger.info(
            f"Embedding retriever loaded: {self.embeddings.shape}, "
            f"{len(self.doc_ids)} docs, {self.embeddings.nbytes / (1024**2):.2f} MB"
        )

    def retrieve(self, query_text: str, top_k: int = None) -> list[dict]:
        if not self.loaded:
            self.load()

        if top_k is None:
            top_k = get_config("retrieval.candidates_k", 1000)

        top_k = min(top_k, len(self.doc_ids))

        if not query_text or not query_text.strip():
            logger.warning("Empty query text for embedding retrieval")
            raise InvalidQueryError("Query text cannot be empty for embedding retrieval")

        # Apply the same preprocessing to query as documents
        from services.preprocessing.preprocessing_service import preprocess
        processed_tokens = preprocess(query_text, mode="query")
        
        processed_query = " ".join(processed_tokens) if processed_tokens else query_text
        
        if not processed_query.strip():
            logger.warning("Query preprocessing resulted in empty text, using original")
            processed_query = query_text
        
        logger.debug(f"Original query: '{query_text[:50]}...'")
        logger.debug(f"Preprocessed query: '{processed_query[:50]}...'")

        normalize = get_config("models.embedding.normalize", True)

        query_embedding = self.model.encode(
            [processed_query],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            device=self.device
        )

        if query_embedding.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[1]} "
                f"doesn't match index dimension {self.embeddings.shape[1]}"
            )

        scores = (self.embeddings @ query_embedding.T).flatten()

        if not np.all(np.isfinite(scores)):
            logger.error("Computed scores contain NaN or Inf values")
            raise ValueError("Invalid similarity scores computed")

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = [
            {"doc_id": self.doc_ids[i], "score": float(scores[i])}
            for i in top_idx
        ]

        logger.info(
            f"Embedding retriever returned {len(results)} results "
            f"(top score: {results[0]['score']:.4f})" if results else "0 results"
        )

        return results

    def get_statistics(self) -> dict:
        if not self.loaded:
            self.load()

        from common.model_loader import get_embedding_model_config
        model_config = get_embedding_model_config()

        return {
            "total_docs": len(self.doc_ids),
            "embedding_dimension": self.embeddings.shape[1],
            "memory_mb": self.embeddings.nbytes / (1024**2),
            "device": self.device,
            "model_name": model_config["model_name"],
            "local_path": model_config["local_path"]
        }
    
    def score_documents(self, query_text: str, doc_ids: list[str]) -> dict[str, float]:
        if not self.loaded:
            self.load()
        
        if not query_text or not query_text.strip() or not doc_ids:
            return {}
        
        from services.preprocessing.preprocessing_service import preprocess
        processed_tokens = preprocess(query_text, mode="query")
        processed_query = " ".join(processed_tokens) if processed_tokens else query_text
        
        if not processed_query.strip():
            processed_query = query_text
        
        normalize = get_config("models.embedding.normalize", True)
        
        query_embedding = self.model.encode(
            [processed_query],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            device=self.device
        )
        
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
        
        doc_embeddings = self.embeddings[doc_indices]
        
        scores = (doc_embeddings @ query_embedding.T).flatten()
        
        result = {valid_doc_ids[i]: float(scores[i]) for i in range(len(valid_doc_ids))}
        
        for doc_id in doc_ids:
            if doc_id not in result:
                result[doc_id] = 0.0
        
        logger.debug(f"Scored {len(result)} documents")
        
        return result
