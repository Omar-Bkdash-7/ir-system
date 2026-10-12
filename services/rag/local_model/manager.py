"""Local model lifecycle: detect -> download (resumable, verified, atomic) -> report status.

Design rules (each one prevents a real failure mode):
* Files are downloaded into "<models_dir>/<key>.partial" and only renamed to "<models_dir>/<key>" after
  verification. "ready" therefore NEVER means "half downloaded".
* Readiness is derived from the files (config + tokenizer + COMPLETE safetensors), not from a flag, so a model
  copied by hand (docker cp) is detected automatically.
* The transfer is done by local_model/downloader.py (plain HTTPS, exact resume, SHA-256, stall detection). It
  runs in a DAEMON thread: a uvicorn --reload or a container stop never waits for a 1 GB transfer, and the next
  start simply resumes from the bytes on disk.
* Downloads are user-triggered. The only automatic action is resuming a download the user already started
  (an "intent" file) after the service was restarted.
* One download at a time. Run the service with a single uvicorn worker.
"""
from __future__ import annotations

import asyncio
import errno
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from common.logging_config import setup_logging
from services.rag.config import RAGSettings
from services.rag.local_model.downloader import (
    MANIFEST_FILE,
    DownloadAborted,
    DownloadError,
    ModelDownloader,
    ProgressTracker,
)
from services.rag.local_model.registry import LocalModelSpec, get_spec
from services.rag.netutil import has_dns_cause

logger = setup_logging("rag.local_model")

MANUAL_HINT = (
    "Manual alternative: download the model on the host (hf download <repo> --local-dir <dir>) and copy it with "
    "'docker cp <dir>/. ir_rag:/models/rag/<key>/' - it is detected automatically."
)


# --------------------------------------------------------------------------- helpers
def safetensors_is_complete(path: Path) -> bool:
    """A .safetensors file starts with a JSON header that declares where every tensor ends.
    If the file is shorter than that, the download was truncated."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                return False
            header_len = int.from_bytes(raw, "little")
            if header_len <= 0 or header_len > 100 * 1024 * 1024 or 8 + header_len > size:
                return False
            header = json.loads(fh.read(header_len))
        end = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            end = max(end, int(info["data_offsets"][1]))
        return size >= 8 + header_len + end
    except Exception:
        return False


def model_dir_is_complete(directory: Path) -> bool:
    if not directory.is_dir() or not (directory / "config.json").is_file():
        return False
    has_tokenizer = (
        (directory / "tokenizer.json").is_file()
        or (directory / "tokenizer.model").is_file()
        or ((directory / "vocab.json").is_file() and (directory / "merges.txt").is_file())
    )
    if not has_tokenizer:
        return False
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        return False
    index = directory / "model.safetensors.index.json"
    if index.is_file():  # sharded checkpoint: every listed shard must exist
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        except Exception:
            return False
        if not shards or not all((directory / s).is_file() for s in shards):
            return False
    return all(safetensors_is_complete(w) for w in weights)


def classify_download_error(exc: BaseException) -> Dict[str, Optional[str]]:
    """Fallback classification for UNEXPECTED exceptions (the downloader raises typed errors itself)."""
    name = type(exc).__name__
    if isinstance(exc, PermissionError):
        return {"code": "permission_denied", "message": f"Cannot write to the models volume: {exc}",
                "hint": "Make sure /models is owned by uid 1000 (see Dockerfile) or recreate the rag_models volume."}
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return {"code": "disk_full", "message": "No space left on device.", "hint": "Free disk space and retry."}
    if has_dns_cause(exc):
        return {"code": "dns_error", "message": "Cannot resolve the download host from the container.", "hint": MANUAL_HINT}
    return {"code": "download_failed", "message": f"{name}: {exc}"[:300], "hint": MANUAL_HINT}


# --------------------------------------------------------------------------- manager
class LocalModelManager:
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.spec: Optional[LocalModelSpec] = get_spec(settings.local_model)
        if self.spec is None:
            logger.error("Unknown RAG_LOCAL_MODEL=%r - local model disabled", settings.local_model)
        key = self.spec.key if self.spec else "unknown"
        self.model_dir: Path = settings.models_dir / key
        self.partial_dir: Path = settings.models_dir / f"{key}.partial"
        self.intent_file: Path = settings.models_dir / f"{key}.intent"  # "the user started a download"
        self.loaded = False  # set by LocalGenerator once the weights are in RAM
        self._state = "not_downloaded"
        self._error: Optional[Dict[str, Optional[str]]] = None
        self._lock = threading.RLock()
        self._progress = ProgressTracker()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_refresh = 0.0
        self.refresh_state(force=True)

    # ---- state ---------------------------------------------------------------
    @property
    def state(self) -> str:
        if self.spec is None or not self.settings.local_enabled:
            return "disabled"
        self.refresh_state()
        return self._state

    def refresh_state(self, force: bool = False) -> None:
        with self._lock:
            if self.spec is None or self._state == "downloading":
                return
            if not force and self._state == "ready":
                return
            now = time.monotonic()
            if not force and now - self._last_refresh < 2.0:
                return
            self._last_refresh = now
            if model_dir_is_complete(self.model_dir):
                self._state, self._error = "ready", None
            elif self._state != "error":
                self._state = "not_downloaded"

    def status(self) -> Dict[str, Any]:
        state = self.state
        snap = self._progress.snapshot()
        busy = state in ("downloading", "error")
        total = snap["total_bytes"] if busy else None
        done = int(snap["downloaded_bytes"]) if state == "downloading" else 0
        if state == "ready":
            progress = 1.0
        elif state == "downloading" and total:
            progress = min(done / total, 0.99)
        else:
            progress = 0.0
        spec = self.spec
        downloading = state == "downloading"
        return {
            "id": spec.key if spec else self.settings.local_model,
            "repo_id": spec.repo_id if spec else None,
            "name": spec.display_name if spec else None,
            "type": "local",
            "state": state,
            "loaded": self.loaded,
            "progress": round(progress, 4),
            "downloaded_bytes": done,
            "total_bytes": total,
            "approx_size_mb": spec.approx_size_mb if spec else None,
            "error": self._error if state == "error" else None,
            "available": state == "ready",
            # live details (stable keys, neutral values when nothing is downloading)
            "phase": snap["phase"] if downloading else "idle",
            "speed_bps": snap["speed_bps"] if downloading else 0,
            "eta_s": snap["eta_s"] if downloading else None,
            "current_file": snap["current_file"] if downloading else None,
            "files_done": snap["files_done"] if busy else 0,
            "files_total": snap["files_total"] if busy else 0,
            "failures": snap["failures"] if downloading else 0,
            "max_failures": snap["max_failures"] if downloading else 0,
            "retry_in_s": snap["retry_in_s"] if downloading else 0,
            "stalled_for_s": snap["stalled_for_s"] if downloading else None,
            "slow": bool(snap["slow"]) if downloading else False,
            "last_error": snap["last_error"] if downloading else None,
            "endpoint": snap["endpoint"] if downloading else None,
        }

    # ---- download ------------------------------------------------------------
    async def start_download(self) -> Dict[str, Any]:
        """Idempotent. Returns immediately; the download runs in a background daemon thread."""
        with self._lock:
            self.refresh_state(force=True)
            if self.spec is None or not self.settings.local_enabled:
                return self.status()
            if self._state in ("ready", "downloading"):
                return self.status()
            self._spawn()
            return self.status()

    def resume_if_interrupted(self) -> bool:
        """Call once at startup: continue a download the user started before a restart/reload. Never starts a
        NEW download, never retries one that ended in an error (the user decides)."""
        with self._lock:
            if not self.settings.download.auto_resume or self.spec is None or not self.settings.local_enabled:
                return False
            self.refresh_state(force=True)
            if self._state != "not_downloaded" or not self.intent_file.exists() or not self.partial_dir.exists():
                return False
            logger.info("Resuming the download interrupted by a restart")
            self._spawn()
            return True

    def _spawn(self) -> None:
        self._state, self._error = "downloading", None
        self._progress.reset()
        self._progress.update(phase="preparing", max_failures=self.settings.download.max_failures)
        self._cancel = threading.Event()
        try:
            self.settings.models_dir.mkdir(parents=True, exist_ok=True)
            self.intent_file.write_text(str(time.time()), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write the download intent file: %s", exc)
        thread = threading.Thread(target=self._thread_main, args=(self._cancel,), name="rag-model-download", daemon=True)
        self._thread = thread
        thread.start()

    def _thread_main(self, cancel: threading.Event) -> None:
        outcome = "failed"
        try:
            downloader = ModelDownloader(
                self.spec.repo_id, self.partial_dir, self.spec.allow_patterns, self.spec.ignore_patterns,
                self.settings.download, self._progress, cancel,
            )
            downloader.run()
            self._publish()
            with self._lock:
                self._state, self._error = "ready", None
            outcome = "ready"
            logger.info("Local model %s is ready at %s", self.spec.key, self.model_dir)
        except DownloadAborted:
            outcome = "aborted"
            with self._lock:
                self._state = "not_downloaded"  # partial files are kept and resumed next time
            logger.info("Download paused (cancelled); partial files kept for the next resume")
        except DownloadError as exc:
            self._fail(exc.to_dict())
        except Exception as exc:  # a bug or an exotic failure: never let the thread die silently
            logger.exception("Unexpected download failure")
            self._fail(classify_download_error(exc))
        finally:
            self._progress.update(phase="idle")
            if outcome != "aborted":  # keep the intent only when we were interrupted by a shutdown/reload
                try:
                    self.intent_file.unlink(missing_ok=True)
                except OSError:
                    pass

    def _fail(self, error: Dict[str, Optional[str]]) -> None:
        with self._lock:
            self._state, self._error = "error", error
        logger.error("Local model download failed: %s - %s", error.get("code"), error.get("message"))

    def _publish(self) -> None:
        if not model_dir_is_complete(self.partial_dir):
            raise RuntimeError("Downloaded files failed verification (incomplete model).")
        self._progress.update(phase="finalizing")
        (self.partial_dir / MANIFEST_FILE).unlink(missing_ok=True)
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
        os.replace(self.partial_dir, self.model_dir)  # atomic: same filesystem

    async def shutdown(self) -> None:
        """Ask the download to stop at the next chunk. Partial files are kept (and resumed after the restart)."""
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 3.0)
