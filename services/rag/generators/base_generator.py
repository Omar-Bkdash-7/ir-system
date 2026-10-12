"""Common contract for all generators (remote and local)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.rag.schemas import ContextDocument, GenerationConfig


@dataclass
class GenerationRequest:
    query: str
    documents: Sequence[ContextDocument]
    config: GenerationConfig


@dataclass
class GenerationResult:
    text: str
    source: str  # "remote" | "local"
    model: str
    latency_ms: int
    usage: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)


class BaseGenerator(ABC):
    source: str = "base"

    @abstractmethod
    def unavailable_reason(self) -> Optional[str]:
        """Return None when the generator can be tried now, else a short human reason."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a result or raise services.rag.errors.GeneratorError. Never return error text."""
