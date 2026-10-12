"""Routing policy: which generator to try, in which order, and how fallback is reported."""
from __future__ import annotations

from typing import Dict, List, Optional

from common.logging_config import setup_logging
from services.rag.config import RAGSettings
from services.rag.errors import Attempt, GenerationUnavailableError, GeneratorError
from services.rag.generators.base_generator import BaseGenerator, GenerationRequest, GenerationResult
from services.rag.remote_monitor import RemoteMonitor

logger = setup_logging("rag.router")

ORDER = {"auto": ("remote", "local"), "remote": ("remote",), "local": ("local",)}


class GeneratorRouter:
    def __init__(self, settings: RAGSettings, remote: BaseGenerator, local: BaseGenerator, monitor: RemoteMonitor):
        self.settings = settings
        self.generators: Dict[str, BaseGenerator] = {"remote": remote, "local": local}
        self.monitor = monitor

    async def generate(self, request: GenerationRequest, mode: Optional[str] = None) -> GenerationResult:
        mode = mode or request.config.mode or self.settings.default_mode
        order = ORDER.get(mode, ORDER["auto"])
        attempts: List[Attempt] = []

        for source in order:
            generator = self.generators[source]

            reason = generator.unavailable_reason()
            if reason:
                attempts.append(Attempt(source, "skipped", "unavailable", reason))
                continue
            # In auto mode, do not waste time on a remote that just failed; remote-only mode always tries.
            if source == "remote" and mode == "auto" and self.monitor.is_open():
                attempts.append(Attempt(source, "skipped", self.monitor.code or "circuit_open",
                                        f"remote paused after a recent failure; retry in {self.monitor.retry_in()}s"))
                continue

            try:
                result = await generator.generate(request)
            except GeneratorError as err:
                logger.warning("%s generator failed: %s (%s)", source, err.code, err.message)
                if source == "remote":
                    self.monitor.record_failure(err)
                attempts.append(Attempt(source, "failed", err.code, err.message))
                continue

            if source == "remote":
                self.monitor.record_success()
            result.fallback_used = bool(attempts)
            if attempts:
                last = attempts[-1]
                result.fallback_reason = f"{last.source} {last.outcome}: {last.code or ''} {last.message or ''}".strip()
            result.attempts = [a.to_dict() for a in attempts]
            return result

        raise GenerationUnavailableError(attempts)
