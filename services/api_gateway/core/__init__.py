"""Framework-free core of the API gateway: search orchestration, dataset metadata, RAG pass-through."""
from services.api_gateway.core.dataset import dataset_info, fallback_info, preprocessing_config
from services.api_gateway.core.rag import RagProxy
from services.api_gateway.core.search import SearchService

__all__ = ["RagProxy", "SearchService", "dataset_info", "fallback_info", "preprocessing_config"]
