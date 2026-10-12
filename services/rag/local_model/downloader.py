"""Resumable, verified, cancellable model-file downloader over plain HTTPS (httpx only).

Why this exists instead of huggingface_hub.snapshot_download:
* The Xet backend (hf_xet, used by default since huggingface_hub 0.32) does NOT resume an `.incomplete` file
  (no `resume_size` is passed to it), keeps the data it fetches in a separate chunk cache and assembles the file
  IN ORDER. The file therefore looks frozen while the network is busy, and its internal retry policy can back off
  for a very long time. It is a known source of stalls on some networks (Docker Desktop / WSL2 NAT, some CDNs).
* The classic hub path writes in 10 MB steps, cannot be cancelled, cannot be watched, and only checks the size.

This module needs nothing but httpx and the standard library: exact resume with `Range`, 1 MB progress
granularity, stall and slow-connection detection, retries with backoff that reset on progress, SHA-256
verification, instant cancellation, optional mirrors. It never imports huggingface_hub.
"""
from __future__ import annotations

import errno
import fnmatch
import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple, TypeVar
from urllib.parse import quote

import httpx

from common.logging_config import setup_logging
from services.rag.config import DownloadSettings
from services.rag.netutil import has_dns_cause

logger = setup_logging("rag.downloader")

T = TypeVar("T")

USER_AGENT = "ir-rag-model-downloader/1.0"
PART_SUFFIX = ".part"
REVISION_FILE = ".revision"
MANIFEST_FILE = ".manifest.json"
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


# --------------------------------------------------------------------------------------- errors
class DownloadError(Exception):
    def __init__(self, code: str, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


class TransientDownloadError(DownloadError):
    """Worth retrying (network hiccup, 5xx, stall). The retry resumes from the bytes already on disk."""

    def __init__(self, code: str, message: str, hint: Optional[str] = None, *,
                 retry_after: Optional[float] = None, reset_partial: bool = False):
        super().__init__(code, message, hint)
        self.retry_after = retry_after
        self.reset_partial = reset_partial  # the partial file cannot be trusted: restart this file from zero


class FatalDownloadError(DownloadError):
    """Not worth retrying (disk full, file missing, retry budget exhausted)."""


class DownloadAborted(Exception):
    """Cancellation requested (service shutting down). Partial files stay on disk for the next resume."""


# --------------------------------------------------------------------------------------- progress
class ProgressTracker:
    """Thread-safe progress shared by the downloader thread and the API (`status()`)."""

    SPEED_WINDOW_S = 8.0

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._lock = threading.Lock()
        self.clock = clock
        self._init_fields()

    def _init_fields(self) -> None:
        self.phase = "idle"  # idle | preparing | downloading | retrying | verifying | finalizing
        self.total: Optional[int] = None
        self.done = 0  # bytes of the model that are on disk (can go down if a file must be redone)
        self.received = 0  # bytes received from the network (never goes down: used to detect progress)
        self.files_total = 0
        self.files_done = 0
        self.current_file: Optional[str] = None
        self.failures = 0
        self.max_failures = 0
        self.retry_at: Optional[float] = None
        self.last_error: Optional[Dict[str, Optional[str]]] = None
        self.endpoint: Optional[str] = None
        self.slow = False
        self.last_progress_at: Optional[float] = None
        self._samples: Deque[Tuple[float, int]] = deque(maxlen=512)

    def reset(self) -> None:
        with self._lock:
            self._init_fields()

    def update(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                if not hasattr(self, key):
                    raise AttributeError(f"ProgressTracker has no field '{key}'")
                setattr(self, key, value)

    def add(self, n: int) -> None:
        with self._lock:
            now = self.clock()
            self.done += n
            self.received += n
            self.last_progress_at = now
            self._samples.append((now, self.received))

    def set_done(self, n: int) -> None:
        with self._lock:
            self.done = n

    def schedule_retry(self, delay: float) -> None:
        with self._lock:
            self.phase = "retrying"
            self.retry_at = self.clock() + delay

    def clear_retry(self) -> None:
        with self._lock:
            self.phase = "downloading"
            self.retry_at = None

    @property
    def received_bytes(self) -> int:
        with self._lock:
            return self.received

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            now = self.clock()
            speed = 0.0
            recent = [(t, b) for t, b in self._samples if now - t <= self.SPEED_WINDOW_S]
            if len(recent) >= 2 and recent[-1][0] > recent[0][0] and now - recent[-1][0] <= 3.0:
                speed = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
            remaining = (self.total - self.done) if self.total else None
            eta = int(remaining / speed) if (remaining is not None and speed > 1024) else None
            stalled = None
            if self.phase == "downloading" and self.last_progress_at is not None:
                stalled = round(now - self.last_progress_at, 1)
            return {
                "phase": self.phase,
                "total_bytes": self.total,
                "downloaded_bytes": self.done,
                "speed_bps": int(speed),
                "eta_s": eta,
                "current_file": self.current_file,
                "files_done": self.files_done,
                "files_total": self.files_total,
                "failures": self.failures,
                "max_failures": self.max_failures,
                "retry_in_s": max(0, int(self.retry_at - now)) if self.retry_at else 0,
                "stalled_for_s": stalled,
                "slow": self.slow,
                "last_error": self.last_error,
                "endpoint": self.endpoint,
            }


# --------------------------------------------------------------------------------------- manifest
@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int
    sha256: Optional[str] = None  # only for LFS/Xet files (the Hub publishes it); small files are checked by size


@dataclass(frozen=True)
class Manifest:
    repo_id: str
    revision: str
    files: Tuple[RemoteFile, ...]

    def to_json(self) -> str:
        return json.dumps({"repo_id": self.repo_id, "revision": self.revision,
                           "files": [{"name": f.name, "size": f.size, "sha256": f.sha256} for f in self.files]})

    @classmethod
    def from_json(cls, raw: str) -> "Manifest":
        data = json.loads(raw)
        return cls(data["repo_id"], data["revision"],
                   tuple(RemoteFile(f["name"], int(f["size"]), f.get("sha256")) for f in data["files"]))


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def remove_legacy_artifacts(partial_dir: Path) -> int:
    """Delete what the old huggingface_hub/Xet-based downloader left behind. Returns the freed bytes.
    * <partial_dir>/.cache : hub bookkeeping + `.incomplete` files whose content cannot be trusted
    * $HF_HOME/xet        : the Xet chunk cache (only when HF_HOME is explicitly set, i.e. inside the container)"""
    freed = 0
    targets = [partial_dir / ".cache"]
    if os.environ.get("HF_HOME"):
        targets.append(Path(os.environ["HF_HOME"]) / "xet")
    for target in targets:
        if target.is_dir():
            freed += _dir_size(target)
            shutil.rmtree(target, ignore_errors=True)
    if freed:
        logger.info("Removed %d MB of leftovers from the previous downloader", freed // (1024 * 1024))
    return freed


# --------------------------------------------------------------------------------------- downloader
class ModelDownloader:
    """Blocking. Run it in a (daemon) thread. Raises FatalDownloadError / DownloadAborted, never Transient."""

    def __init__(self, repo_id: str, dest: Path, allow_patterns: Sequence[str], ignore_patterns: Sequence[str],
                 cfg: DownloadSettings, progress: ProgressTracker, cancel: threading.Event,
                 transport: Optional[httpx.BaseTransport] = None):
        self.repo_id = repo_id
        self.dest = Path(dest)
        self.allow = tuple(allow_patterns)
        self.ignore = tuple(ignore_patterns)
        self.cfg = cfg
        self.progress = progress
        self.cancel = cancel
        self._transport = transport
        self._endpoints: List[str] = [cfg.hf_endpoint, *cfg.fallback_endpoints]
        self._ep = 0
        self._slow_restarts = 0
        self._best_done = 0
        self._files: Tuple[RemoteFile, ...] = ()

    # ---- public --------------------------------------------------------------------------
    @property
    def endpoint(self) -> str:
        return self._endpoints[self._ep]

    def run(self) -> None:
        p = self.progress
        p.update(phase="preparing", max_failures=self.cfg.max_failures, endpoint=self.endpoint)
        manifest = self._load_manifest()
        self._prepare_dir(manifest)
        self._files = tuple(sorted(manifest.files, key=lambda f: f.size))  # small files first, the big one last
        p.update(total=sum(f.size for f in self._files), files_total=len(self._files), files_done=0)
        self._resync_done()
        self._check_disk()
        logger.info("Download start: repo=%s revision=%s files=%d total=%d MB already_on_disk=%d MB",
                    self.repo_id, manifest.revision[:10], len(self._files), (p.total or 0) // 2**20, p.done // 2**20)
        for file in self._files:
            self._raise_if_cancelled()
            p.update(current_file=file.name)
            self._ensure_file(file, manifest.revision)
            p.update(files_done=p.files_done + 1)
        p.update(phase="finalizing", current_file=None)

    # ---- manifest ------------------------------------------------------------------------
    def _load_manifest(self) -> Manifest:
        def attempt() -> Manifest:
            last: Optional[DownloadError] = None
            for endpoint in self._endpoints:  # primary first: hashes should come from the official host if reachable
                try:
                    return self._fetch_manifest(endpoint)
                except TransientDownloadError as exc:
                    last = exc
            saved = self._read_saved_manifest()
            if saved is not None:
                logger.warning("API unreachable: using the manifest saved by the previous attempt")
                return saved
            assert last is not None
            raise last

        manifest = self._retrying("manifest", attempt)
        self.dest.mkdir(parents=True, exist_ok=True)
        (self.dest / MANIFEST_FILE).write_text(manifest.to_json(), encoding="utf-8")
        return manifest

    def _fetch_manifest(self, endpoint: str) -> Manifest:
        url = f"{endpoint}/api/models/{self.repo_id}/revision/main"
        try:
            with self._client() as client:
                response = client.get(url, params={"blobs": "true"})
        except httpx.HTTPError as exc:
            raise self._translate_http_error(exc) from exc
        if response.status_code == 404:
            raise FatalDownloadError("not_found", f"Model repository '{self.repo_id}' was not found on {endpoint}.")
        if response.status_code in (401, 403):
            raise FatalDownloadError("forbidden", f"{endpoint} refused access to '{self.repo_id}' (HTTP {response.status_code}).")
        if response.status_code != 200:
            raise self._status_error(response)
        try:
            data = response.json()
            revision = data["sha"]
            files = tuple(
                RemoteFile(s["rfilename"], int(s["size"]), (s.get("lfs") or {}).get("sha256"))
                for s in data.get("siblings", [])
                if s.get("rfilename") and s.get("size") is not None and self._wanted(s["rfilename"])
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise TransientDownloadError("bad_manifest", f"Unexpected API answer from {endpoint}: {exc}") from exc
        if not files:
            raise FatalDownloadError("bad_manifest", f"No downloadable files match the patterns in '{self.repo_id}'.")
        return Manifest(self.repo_id, revision, files)

    def _read_saved_manifest(self) -> Optional[Manifest]:
        try:
            return Manifest.from_json((self.dest / MANIFEST_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            return None

    def _wanted(self, name: str) -> bool:
        return (any(fnmatch.fnmatch(name, p) for p in self.allow)
                and not any(fnmatch.fnmatch(name, p) for p in self.ignore))

    # ---- directory state --------------------------------------------------------------------
    def _prepare_dir(self, manifest: Manifest) -> None:
        self.dest.mkdir(parents=True, exist_ok=True)
        remove_legacy_artifacts(self.dest)
        marker = self.dest / REVISION_FILE
        if marker.exists() and marker.read_text(encoding="utf-8").strip() != manifest.revision:
            logger.warning("The model changed upstream: discarding the old partial download")
            shutil.rmtree(self.dest, ignore_errors=True)
            self.dest.mkdir(parents=True, exist_ok=True)
            (self.dest / MANIFEST_FILE).write_text(manifest.to_json(), encoding="utf-8")
        marker.write_text(manifest.revision, encoding="utf-8")

    def _part_path(self, file: RemoteFile) -> Path:
        return self.dest / (file.name + PART_SUFFIX)

    def _bytes_present(self) -> int:
        total = 0
        for file in self._files:
            final = self.dest / file.name
            if final.is_file() and final.stat().st_size == file.size:
                total += file.size
            else:
                part = self._part_path(file)
                if part.is_file():
                    total += min(part.stat().st_size, file.size)
        return total

    def _resync_done(self) -> None:
        self.progress.set_done(self._bytes_present())

    def _check_disk(self) -> None:
        needed = (self.progress.total or 0) - self.progress.done
        free = shutil.disk_usage(self.dest).free
        if free < int(needed * 1.1) + 64 * 1024 * 1024:
            raise FatalDownloadError(
                "disk_full", f"Not enough free disk space: need ~{needed // 2**20} MB more, have {free // 2**20} MB.",
                "Free disk space in Docker Desktop (Settings > Resources) and retry.")

    # ---- per-file logic --------------------------------------------------------------------
    def _ensure_file(self, file: RemoteFile, revision: str) -> None:
        final, part = self.dest / file.name, self._part_path(file)
        if final.is_file() and final.stat().st_size == file.size:
            return  # downloaded and verified by a previous attempt (we verify BEFORE the rename)
        final.unlink(missing_ok=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        integrity_failures = 0
        while True:
            self._retrying(f"download {file.name}", lambda: self._download_attempt(file, revision, part))
            if file.sha256:
                self.progress.update(phase="verifying")
                actual = sha256_of(part)
                if actual != file.sha256:
                    integrity_failures += 1
                    logger.error("SHA-256 mismatch for %s (got %s, expected %s)", file.name, actual[:12], file.sha256[:12])
                    part.unlink(missing_ok=True)
                    self._resync_done()
                    if integrity_failures >= 2:
                        raise FatalDownloadError(
                            "integrity_error", f"{file.name} failed SHA-256 verification twice: the server sent corrupted data.",
                            "Try again later, or configure RAG_HF_FALLBACK_ENDPOINTS.")
                    self.progress.update(phase="downloading")
                    continue
            os.replace(part, final)  # atomic: a file under its final name is always complete and verified
            self.progress.update(phase="downloading")
            logger.info("File ready: %s (%d bytes)", file.name, file.size)
            return

    def _download_attempt(self, file: RemoteFile, revision: str, part: Path) -> None:
        """One HTTP request. Resumes from the bytes already in `part`. Raises Transient/Fatal/DownloadAborted."""
        offset = part.stat().st_size if part.is_file() else 0
        if offset > file.size:
            self._drop_partial(part)
            offset = 0
        if offset == file.size:
            return  # complete already (the previous attempt was killed before verification/rename)
        headers = {"Accept-Encoding": "identity", "User-Agent": USER_AGENT}  # identity: byte ranges must be raw bytes
        if offset:
            headers["Range"] = f"bytes={offset}-"
            logger.info("Resuming %s from byte %d of %d", file.name, offset, file.size)
        self.progress.update(phase="downloading", endpoint=self.endpoint)
        try:
            with self._client() as client, client.stream("GET", self._file_url(revision, file.name), headers=headers) as response:
                mode = self._check_response(response, offset, file)
                if mode == "restart":  # the server ignored our Range header: rewrite the file from the start
                    offset = 0
                    self._drop_partial(part)
                part.parent.mkdir(parents=True, exist_ok=True)
                with part.open("ab" if offset else "wb") as fh:
                    self._copy(response, fh, file, offset)
        except TransientDownloadError as exc:
            if exc.reset_partial:
                self._drop_partial(part)
            raise
        except (DownloadError, DownloadAborted):
            raise
        except httpx.HTTPError as exc:
            raise self._translate_http_error(exc) from exc
        except OSError as exc:
            raise self._translate_os_error(exc) from exc

    def _check_response(self, response: httpx.Response, offset: int, file: RemoteFile) -> str:
        status = response.status_code
        if status == 206:
            match = _CONTENT_RANGE.match(response.headers.get("content-range", ""))
            if not match or int(match.group(1)) != offset:
                raise TransientDownloadError("range_mismatch", "The server answered a different byte range than requested.",
                                             reset_partial=True)
            if match.group(3) != "*" and int(match.group(3)) != file.size:
                raise TransientDownloadError("size_mismatch", f"{file.name} has a different size on the server.", reset_partial=True)
            return "append"
        if status == 200:
            length = response.headers.get("content-length")
            if length and int(length) != file.size:
                raise TransientDownloadError("size_mismatch", f"{file.name} has a different size on the server.", reset_partial=True)
            return "restart" if offset else "append"
        if status == 416:
            raise TransientDownloadError("range_not_satisfiable", "The server rejected the resume position.", reset_partial=True)
        if status == 404:
            raise FatalDownloadError("not_found", f"{file.name} was not found on the server (HTTP 404).")
        raise self._status_error(response)

    def _copy(self, response: httpx.Response, fh, file: RemoteFile, offset: int) -> None:
        cfg, p = self.cfg, self.progress
        written = offset
        window_start: Optional[float] = None
        window_bytes = 0
        for chunk in response.iter_raw(cfg.chunk_bytes):
            if self.cancel.is_set():
                raise DownloadAborted()
            if not chunk:
                continue
            fh.write(chunk)
            fh.flush()  # the size on disk always equals what we report and what a resume will use
            written += len(chunk)
            p.add(len(chunk))
            if written > file.size:
                raise TransientDownloadError("size_mismatch", f"The server sent more data than expected for {file.name}.",
                                             reset_partial=True)
            if cfg.min_speed_kbps > 0:
                now = time.monotonic()
                if window_start is None:
                    window_start, window_bytes = now, 0
                    continue
                window_bytes += len(chunk)
                if now - window_start >= cfg.speed_window:
                    speed = window_bytes / (now - window_start)
                    if speed < cfg.min_speed_kbps * 1024:
                        if self._slow_restarts < cfg.max_slow_restarts:
                            self._slow_restarts += 1
                            p.update(slow=True)
                            raise TransientDownloadError("slow_connection",
                                                         f"The transfer is too slow ({speed / 1024:.0f} KB/s): reconnecting.")
                        p.update(slow=True)  # accepted: keep going, but tell the user
                    else:
                        self._slow_restarts = 0
                        p.update(slow=False)
                    window_start, window_bytes = now, 0
        fh.flush()
        os.fsync(fh.fileno())
        if written != file.size:
            raise TransientDownloadError("incomplete_response", f"The connection ended after {written} of {file.size} bytes.")

    # ---- retries -----------------------------------------------------------------------------
    def _retrying(self, label: str, attempt: Callable[[], T]) -> T:
        """Run `attempt`; on a transient error back off and call it again. The failure budget is only consumed by
        attempts that received NO new byte, so a flaky-but-moving connection always finishes."""
        failures = 0
        while True:
            self._raise_if_cancelled()
            before = max(self._best_done, self.progress.done)
            try:
                return attempt()
            except TransientDownloadError as exc:
                after = self.progress.done
                progressed = after > before
                self._best_done = max(self._best_done, after)
                failures = 0 if progressed else failures + 1
                self.progress.update(failures=failures, last_error=exc.to_dict())
                if failures >= self.cfg.max_failures:
                    raise FatalDownloadError(exc.code, f"{exc.message} (gave up after {failures} attempts without progress)",
                                             exc.hint) from exc
                self._maybe_rotate_endpoint(failures)
                delay = self._backoff(failures, exc.retry_after)
                logger.warning("%s: %s [%s] - %s; retry in %.1fs (failures without progress: %d/%d)",
                               label, exc.code, "progress made" if progressed else "no progress", exc.message, delay,
                               failures, self.cfg.max_failures)
                self._sleep(delay)

    def _backoff(self, failures: int, retry_after: Optional[float]) -> float:
        if failures == 0:  # we made progress: reconnect quickly
            delay = min(1.0, self.cfg.backoff_base)
        else:
            delay = min(self.cfg.backoff_max, self.cfg.backoff_base * (2 ** (failures - 1)))
        delay *= 0.75 + 0.5 * random.random()  # jitter
        if retry_after:
            delay = max(delay, min(retry_after, 120.0))
        return delay

    def _sleep(self, delay: float) -> None:
        self.progress.schedule_retry(delay)
        if self.cancel.wait(delay):
            raise DownloadAborted()
        self.progress.clear_retry()

    def _maybe_rotate_endpoint(self, failures: int) -> None:
        if len(self._endpoints) > 1 and failures > 0 and failures % 3 == 0:
            self._ep = (self._ep + 1) % len(self._endpoints)
            logger.warning("Switching download endpoint to %s", self.endpoint)
            self.progress.update(endpoint=self.endpoint)

    def _raise_if_cancelled(self) -> None:
        if self.cancel.is_set():
            raise DownloadAborted()

    # ---- http helpers ------------------------------------------------------------------------
    def _client(self) -> httpx.Client:
        return httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=self.cfg.connect_timeout, read=self.cfg.read_timeout, write=15.0, pool=15.0),
            transport=self._transport,
        )

    def _file_url(self, revision: str, name: str) -> str:
        return f"{self.endpoint}/{self.repo_id}/resolve/{revision}/{quote(name)}"

    def _status_error(self, response: httpx.Response) -> TransientDownloadError:
        status = response.status_code
        retry_after: Optional[float] = None
        try:
            retry_after = float(response.headers.get("retry-after", ""))
        except ValueError:
            pass
        if status == 429:
            return TransientDownloadError("rate_limited", "The server is rate limiting this download (HTTP 429).", retry_after=retry_after)
        if status in (401, 403):
            return TransientDownloadError("forbidden", f"The download link was refused (HTTP {status}).", retry_after=retry_after)
        if status >= 500:
            return TransientDownloadError("server_error", f"The server failed (HTTP {status}).", retry_after=retry_after)
        return TransientDownloadError("unexpected_status", f"Unexpected HTTP {status}.", retry_after=retry_after)

    def _translate_http_error(self, exc: BaseException) -> TransientDownloadError:
        if isinstance(exc, httpx.TimeoutException):
            return TransientDownloadError("timeout", f"The server did not answer within {self.cfg.read_timeout:.0f}s.")
        if isinstance(exc, httpx.ConnectError):
            if has_dns_cause(exc):
                return TransientDownloadError("dns_error", "Cannot resolve the download host (DNS).",
                                              "Check the container's DNS/VPN, or set RAG_HF_ENDPOINT to a reachable mirror.")
            return TransientDownloadError("connect_error", "Cannot connect to the download host.")
        if isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)):
            return TransientDownloadError("connection_dropped", "The connection was closed before the download finished.")
        if isinstance(exc, httpx.TooManyRedirects):
            return TransientDownloadError("redirect_loop", "Too many redirects.")
        return TransientDownloadError("http_error", f"{type(exc).__name__}: {exc}"[:200])

    def _translate_os_error(self, exc: OSError) -> DownloadError:
        if isinstance(exc, PermissionError):
            return FatalDownloadError("permission_denied", f"Cannot write to the models volume: {exc}",
                                      "Make sure /models is owned by uid 1000 (see the Dockerfile) or recreate the rag_models volume.")
        if exc.errno == errno.ENOSPC:
            return FatalDownloadError("disk_full", "No space left on device.", "Free disk space and retry.")
        return FatalDownloadError("io_error", f"{type(exc).__name__}: {exc}"[:300])

    def _drop_partial(self, part: Path) -> None:
        part.unlink(missing_ok=True)
        self._resync_done()
