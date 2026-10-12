"""Central, typed configuration for the RAG service (stdlib only, no pydantic-settings).

Every tunable value is read from environment variables ONCE (see RAGSettings.from_env)
and passed around explicitly. No module should call os.getenv() on its own.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

VALID_MODES = ("auto", "remote", "local")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    return raw in ("1", "true", "yes", "on")


def _env_list(name: str) -> Tuple[str, ...]:
    return tuple(p.strip().rstrip("/") for p in _env(name).split(",") if p.strip())


def _first_env(*names: str) -> str:
    for n in names:
        v = _env(n)
        if v:
            return v
    return ""


@dataclass(frozen=True)
class DownloadSettings:
    """Local-model downloader policy (see local_model/downloader.py)."""

    hf_endpoint: str = "https://huggingface.co"
    fallback_endpoints: Tuple[str, ...] = ()  # optional mirrors, tried only after repeated failures
    connect_timeout: float = 15.0
    read_timeout: float = 30.0       # no bytes for this long = stalled -> reconnect and resume
    max_failures: int = 6            # consecutive attempts WITHOUT progress before giving up
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    min_speed_kbps: float = 64.0     # 0 disables the slow-connection restart
    speed_window: float = 30.0
    max_slow_restarts: int = 3       # after that a slow transfer is accepted (and flagged) instead of restarted
    chunk_bytes: int = 1024 * 1024   # progress + resume granularity
    auto_resume: bool = True         # resume a download the user started if the service restarted


@dataclass(frozen=True)
class RAGSettings:
    # --- retrieval service -------------------------------------------------
    retrieval_url: str = "http://retrieval:8003"
    retrieval_timeout: float = 120.0

    # --- generation policy -------------------------------------------------
    default_mode: str = "auto"  # auto = remote first, local fallback

    # --- remote (any OpenAI-compatible endpoint; default = Hugging Face router)
    remote_base_url: str = "https://router.huggingface.co/v1"
    remote_model: str = "meta-llama/Llama-3.1-8B-Instruct:cheapest"
    remote_api_key: str = field(default="", repr=False)  # never printed
    remote_connect_timeout: float = 5.0
    remote_read_timeout: float = 60.0
    remote_max_retries: int = 1
    remote_max_context_chars: int = 12000

    # --- remote health / circuit breaker -----------------------------------
    cooldown_transient: float = 30.0   # dns / connect / timeout / 5xx / 429
    cooldown_persistent: float = 300.0  # auth / quota / model removed
    probe_ttl_ok: float = 300.0
    probe_ttl_fail: float = 20.0

    # --- local model --------------------------------------------------------
    local_enabled: bool = True
    local_model: str = "qwen2.5-0.5b-instruct"  # key in local_model/registry.py
    models_dir: Path = Path("/models/rag")
    local_threads: int = 2
    local_dtype: str = "float32"  # float32 | bfloat16 | float16
    local_max_new_tokens: int = 256
    local_max_context_chars: int = 5000
    local_max_input_tokens: int = 2048
    local_max_seconds: float = 120.0

    download: DownloadSettings = field(default_factory=DownloadSettings)

    # --- misc ---------------------------------------------------------------
    diagnostics_enabled: bool = False

    @property
    def remote_configured(self) -> bool:
        return bool(self.remote_api_key)

    @property
    def remote_host(self) -> str:
        return urlparse(self.remote_base_url).hostname or self.remote_base_url

    @property
    def remote_model_id(self) -> str:
        """Model id without the router policy suffix (':cheapest', ':fastest', ':<provider>')."""
        return self.remote_model.split(":", 1)[0]

    @classmethod
    def from_env(cls) -> "RAGSettings":
        mode = _env("RAG_DEFAULT_MODE", "auto").lower()
        if mode not in VALID_MODES:
            mode = "auto"
        dtype = _env("RAG_LOCAL_DTYPE", "float32").lower()
        if dtype not in ("float32", "bfloat16", "float16"):
            dtype = "float32"
        return cls(
            retrieval_url=_env("RETRIEVAL_URL", "http://retrieval:8003").rstrip("/"),
            retrieval_timeout=_env_float("RAG_RETRIEVAL_TIMEOUT", 120.0),
            default_mode=mode,
            remote_base_url=_env("RAG_REMOTE_BASE_URL", "https://router.huggingface.co/v1").rstrip("/"),
            remote_model=_env("RAG_REMOTE_MODEL", "meta-llama/Llama-3.1-8B-Instruct:cheapest"),
            remote_api_key=_first_env(
                "RAG_REMOTE_API_KEY", "HUGGINGFACE_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"
            ),
            remote_connect_timeout=_env_float("RAG_REMOTE_CONNECT_TIMEOUT", 5.0),
            remote_read_timeout=_env_float("RAG_REMOTE_READ_TIMEOUT", 60.0),
            remote_max_retries=max(0, _env_int("RAG_REMOTE_MAX_RETRIES", 1)),
            remote_max_context_chars=_env_int("RAG_REMOTE_MAX_CONTEXT_CHARS", 12000),
            cooldown_transient=_env_float("RAG_COOLDOWN_TRANSIENT", 30.0),
            cooldown_persistent=_env_float("RAG_COOLDOWN_PERSISTENT", 300.0),
            local_enabled=_env_bool("RAG_LOCAL_ENABLED", True),
            local_model=_env("RAG_LOCAL_MODEL", "qwen2.5-0.5b-instruct"),
            models_dir=Path(_env("RAG_MODELS_DIR", "/models/rag")),
            local_threads=max(1, _env_int("RAG_LOCAL_THREADS", 2)),
            local_dtype=dtype,
            local_max_new_tokens=_env_int("RAG_LOCAL_MAX_NEW_TOKENS", 256),
            local_max_context_chars=_env_int("RAG_LOCAL_MAX_CONTEXT_CHARS", 5000),
            local_max_input_tokens=_env_int("RAG_LOCAL_MAX_INPUT_TOKENS", 2048),
            local_max_seconds=_env_float("RAG_LOCAL_MAX_SECONDS", 120.0),
            download=DownloadSettings(
                hf_endpoint=_env("RAG_HF_ENDPOINT", "https://huggingface.co").rstrip("/"),
                fallback_endpoints=_env_list("RAG_HF_FALLBACK_ENDPOINTS"),
                connect_timeout=_env_float("RAG_DOWNLOAD_CONNECT_TIMEOUT", 15.0),
                read_timeout=_env_float("RAG_DOWNLOAD_READ_TIMEOUT", 30.0),
                max_failures=max(1, _env_int("RAG_DOWNLOAD_MAX_FAILURES", 6)),
                backoff_base=_env_float("RAG_DOWNLOAD_BACKOFF_BASE", 2.0),
                backoff_max=_env_float("RAG_DOWNLOAD_BACKOFF_MAX", 60.0),
                min_speed_kbps=_env_float("RAG_DOWNLOAD_MIN_SPEED_KBPS", 64.0),
                speed_window=_env_float("RAG_DOWNLOAD_SPEED_WINDOW", 30.0),
                max_slow_restarts=_env_int("RAG_DOWNLOAD_MAX_SLOW_RESTARTS", 3),
                chunk_bytes=max(64 * 1024, _env_int("RAG_DOWNLOAD_CHUNK_BYTES", 1024 * 1024)),
                auto_resume=_env_bool("RAG_DOWNLOAD_AUTO_RESUME", True),
            ),
            diagnostics_enabled=_env_bool("RAG_ENABLE_DIAGNOSTICS", False),
        )
