"""Typed errors for the generation layer.

Generators NEVER return error text as an answer. They raise GeneratorError with a stable
`code`, and the router decides whether to fall back, skip, or give up.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# Errors that will NOT fix themselves within minutes -> long circuit-breaker cooldown.
PERSISTENT_CODES = frozenset({"no_token", "auth_error", "quota_exhausted", "model_unavailable"})


class GeneratorError(Exception):
    def __init__(self, source: str, code: str, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.source = source  # "remote" | "local"
        self.code = code
        self.message = message
        self.hint = hint

    @property
    def persistent(self) -> bool:
        return self.code in PERSISTENT_CODES

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "code": self.code, "message": self.message, "hint": self.hint}


@dataclass
class Attempt:
    """One step of the routing decision (also returned to the UI for transparency)."""

    source: str
    outcome: str  # "ok" | "failed" | "skipped"
    code: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GenerationUnavailableError(Exception):
    """Raised by the router when every allowed generator failed or was unavailable."""

    def __init__(self, attempts: List[Attempt]):
        self.attempts = attempts
        summary = "; ".join(f"{a.source}:{a.code or a.outcome}" for a in attempts) or "no generator allowed"
        super().__init__(f"No generator could produce an answer ({summary})")

    def to_list(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.attempts]
