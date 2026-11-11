"""
Logging and failure wording for the frontend.

Two jobs, kept in one small module so the rest of the app does not grow its own
ad-hoc `print`/`st.error(f"{e}")` handling:

* `get_logger()` — one configured logger tree (`frontend.*`) writing to stderr
  and to `logs/frontend.log`, the same directory the services write to.
* `failure_hint()` — turns the coarse `APIResponse.reason` into a sentence the
  user can act on, instead of repeating "Connection refused" for every cause.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_LOG_DIR = os.environ.get("IR_LOG_DIR", "/app/logs")
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-26s %(message)s"

# Reason -> sentence explaining what the user can actually do about it.
_HINTS = {
    "connection": "the service is not reachable — is its container running?",
    "timeout": "the service did not answer in time; it may still be starting up",
    "server": "the service hit an internal error; check its log",
}

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the `frontend` logger once per process."""
    global _configured
    if _configured:
        return

    logger = logging.getLogger("frontend")
    logger.setLevel(level)
    logger.propagate = False

    # Streamlit re-runs app.py, so make this idempotent per logger instance:
    # without it every import path that calls setup_logging() would append
    # another handler and every line would be written once per handler.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    path = os.path.join(_LOG_DIR, "frontend.log")
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # A read-only log directory must not take the app down.
        logger.warning("cannot write %s; logging to stderr only", path)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Logger for one module, e.g. ``get_logger(__name__)``."""
    setup_logging()
    return logging.getLogger(f"frontend.{name}")


@contextmanager
def timed(logger: logging.Logger, action: str, *, slow_ms: Optional[float] = 1000) -> Iterator[None]:
    """
    Time an action, logging it when it is slow or fails.

    Used around work that is not an API call (parsing, aggregation) so the log
    shows where a slow or broken page actually spent its time.
    """
    started = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("%s failed after %.0fms", action, (time.perf_counter() - started) * 1000)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        if slow_ms is not None and elapsed_ms >= slow_ms:
            logger.info("%s took %.0fms", action, elapsed_ms)


def failure_hint(resp, *, default: str = "unknown error") -> str:
    """
    User-facing description of a failed `APIResponse`.

    The service's own message is kept (it is the most specific thing available)
    and a hint is appended only when the failure class tells the user nothing
    they could not already guess.
    """
    detail = getattr(resp, "error", None) or default
    hint = _HINTS.get(getattr(resp, "reason", None) or "")
    return f"{detail} — {hint}" if hint else detail


def log_failure(logger: logging.Logger, action: str, resp) -> None:
    """Log a failed API call with the fields that make it diagnosable."""
    logger.warning(
        "%s failed: reason=%s status=%s elapsed=%.0fms error=%s",
        action,
        getattr(resp, "reason", None),
        getattr(resp, "status_code", None),
        getattr(resp, "elapsed_ms", 0.0) or 0.0,
        getattr(resp, "error", None),
    )
