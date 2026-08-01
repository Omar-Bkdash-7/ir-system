import numpy as np
import json
import jsonlines
from pathlib import Path
from sentence_transformers import SentenceTransformer
from common.config import get_config
from common.device import get_torch_device
from common.exceptions import IndexNotFoundError, ModelLoadError
import logging

logger = logging.getLogger(__name__)


class EmbeddingIndexer:
    def __init__(self, dataset_name: str = None):
        self.model = None
        self.embeddings = None
        self.doc_ids = []

        # Use dataset-specific paths
        if dataset_name:
            dataset_slug = dataset_name.replace('/', '_').replace('-', '_')
            base_path = Path(get_config("indices.base_path", "data/indices")) / dataset_slug / "embedding"
        else:
            # Fallback to old behavior
            base_path = Path(get_config("indices.base_path", "data/indices"))
        
        self.embeddings_path = base_path / "embeddings.npy"
        self.doc_id_map_path = base_path / "doc_id_map.json"

        self.device = get_torch_device()

    def _load_model(self):
        if self.model is None:
            model_name = get_config("models.embedding.model_name", "models/all-MiniLM-L6-v2")
            logger.info(f"Loading embedding model: {model_name} on {self.device}")

            try:
                self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=False)
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load the model '{model_name}'. "
                    f"Check internet connection or model cache. Original error: {str(e)}"
                )

            expected_dim = get_config("models.embedding.dimension", 384)
            actual_dim = self.model.get_sentence_embedding_dimension()

            if actual_dim != expected_dim:
                logger.warning(
                    f"Model dimension mismatch: expected {expected_dim}, got {actual_dim}. "
                    f"Update config.yaml to dimension: {actual_dim}"
                )

            logger.info(f"Model loaded successfully: dimension={actual_dim}")

    def build(self, processed_path: str = None):
        if processed_path is None:
            processed_path = get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl")

        processed_path = Path(processed_path)
        if not processed_path.exists():
            raise FileNotFoundError(f"Preprocessed documents not found at {processed_path}")

        self._load_model()

        doc_ids = []
        texts = []

        logger.info(f"Loading documents from {processed_path}")

        with jsonlines.open(processed_path) as reader:
            for doc in reader:
                if "doc_id" not in doc or "text" not in doc:
                    logger.warning(f"Skipping malformed document: {doc}")
                    continue

                doc_ids.append(doc["doc_id"])
                texts.append(doc["text"])

        if not texts:
            raise ValueError("No valid documents found to encode")

        logger.info(f"Encoding {len(texts)} documents with sentence-transformers")
        logger.info(f"Device: {self.device}, Batch size: {get_config('models.embedding.batch_size', 64)}")

        batch_size = get_config("models.embedding.batch_size", 64)
        normalize = get_config("models.embedding.normalize", True)

        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            device=self.device
        )

        self.doc_ids = doc_ids

        logger.info(f"Embeddings created successfully")
        logger.info(f"  Shape: {self.embeddings.shape}")
        logger.info(f"  Dtype: {self.embeddings.dtype}")
        logger.info(f"  Memory: {self.embeddings.nbytes / (1024**2):.2f} MB")

        self._validate_embeddings()

        return self

    def _validate_embeddings(self):
        if len(self.doc_ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.doc_ids)} doc_ids but {self.embeddings.shape[0]} embeddings"
            )

        if not np.all(np.isfinite(self.embeddings)):
            raise ValueError("Embeddings contain NaN or Inf values")

        logger.info("Embeddings validation passed")

    def save(self):
        if self.embeddings is None or not self.doc_ids:
            raise ValueError("No embeddings to save. Run build() first.")

        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(self.embeddings_path, self.embeddings)
        logger.info(f"Saved embeddings to {self.embeddings_path}")

        with open(self.doc_id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved doc ID map to {self.doc_id_map_path}")

        self._log_index_info()

    def _log_index_info(self):
        logger.info("Embedding index summary:")
        logger.info(f"  Documents: {len(self.doc_ids)}")
        logger.info(f"  Embedding dimension: {self.embeddings.shape[1]}")
        logger.info(f"  Total size: {self.embeddings.nbytes / (1024**2):.2f} MB")

    def load(self):
        if not self.embeddings_path.exists():
            raise IndexNotFoundError(f"Embeddings not found at {self.embeddings_path}")
        if not self.doc_id_map_path.exists():
            raise IndexNotFoundError(f"Doc ID map not found at {self.doc_id_map_path}")

        logger.info(f"Loading embeddings from {self.embeddings_path}")
        self.embeddings = np.load(self.embeddings_path)

        logger.info(f"Loading doc ID map from {self.doc_id_map_path}")
        with open(self.doc_id_map_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        if len(self.doc_ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.doc_ids)} doc_ids but {self.embeddings.shape[0]} embeddings"
            )

        logger.info(f"Loaded embeddings: {self.embeddings.shape}, {len(self.doc_ids)} docs")
        logger.info(f"Memory usage: {self.embeddings.nbytes / (1024**2):.2f} MB")

        return self

    def exists(self) -> bool:
        return self.embeddings_path.exists() and self.doc_id_map_path.exists()

    def get_embedding_dimension(self) -> int:
        if self.embeddings is not None:
            return self.embeddings.shape[1]
        elif self.exists():
            temp_embeddings = np.load(self.embeddings_path, mmap_mode='r')
            return temp_embeddings.shape[1]
        else:
            return get_config("models.embedding.dimension", 384)
