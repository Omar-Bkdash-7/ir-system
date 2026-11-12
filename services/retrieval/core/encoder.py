"""Query embeddings: the model loads lazily (once), encoding is serialised, and recent query vectors are cached.

The cache matters because the same queries come back all the time (evaluation runs, hybrid requests, the UI re-running
a search), and one CPU encode costs far more than everything else in a request.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import numpy as np

from common.config import get_config
from common.exceptions import ModelLoadError, RetrievalError

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    from common.device import get_torch_device

    return get_torch_device()


def _default_model_factory() -> Any:
    try:
        from common.model_loader import load_embedding_model  # imports sentence-transformers: load it only on demand
    except ImportError as exc:
        raise ModelLoadError(f"sentence-transformers is not available: {exc}") from exc
    return load_embedding_model(device=_resolve_device(), cache=True)


class QueryEncoder:
    def __init__(
        self,
        cache_size: int = 512,
        *,
        model_factory: Optional[Callable[[], Any]] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        self._factory = model_factory or _default_model_factory
        self._device = device
        self._normalize = bool(get_config("models.embedding.normalize", True)) if normalize is None else normalize
        self._cache_size = cache_size
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._encode_lock = threading.Lock()  # one encode at a time: HF fast tokenizers are not safe under concurrent calls

    def _get_model(self) -> Any:
        model = self._model
        if model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._factory()
                model = self._model
        return model

    def _device_name(self) -> Optional[str]:
        if self._device is None:
            try:
                self._device = _resolve_device()
            except Exception:  # device selection must never break a search
                self._device = "cpu"
        return self._device

    def encode(self, text: str) -> np.ndarray:
        """Unit-length float32 vector of the text (read-only; shared between callers)."""
        if self._cache_size:
            with self._cache_lock:
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    return cached
        model = self._get_model()
        try:
            with self._encode_lock:
                embedding = model.encode(
                    [text],
                    normalize_embeddings=self._normalize,
                    convert_to_numpy=True,
                    device=self._device_name(),
                    show_progress_bar=False,
                )
        except ModelLoadError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Query encoding failed: {exc}") from exc
        vector = np.ascontiguousarray(np.asarray(embedding)[0], dtype=np.float32)
        vector.setflags(write=False)
        if self._cache_size:
            with self._cache_lock:
                self._cache[text] = vector
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return vector

    def warm(self) -> None:
        self.encode("warm up")

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def info(self) -> Dict[str, Any]:
        """The embedding half of /status; reads configuration only, never loads the model."""
        return {
            "device": self._device_name(),
            "model_name": get_config("models.embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            "local_path": get_config("models.embedding.local_path", None),
        }
