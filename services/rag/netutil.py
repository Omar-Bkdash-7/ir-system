"""Tiny network helpers shared by the remote generator, the model downloader and diagnostics."""
from __future__ import annotations

import socket
from typing import Optional

_DNS_MARKERS = (
    "name or service not known",
    "no address associated",
    "temporary failure in name resolution",
    "nodename nor servname",
    "failed to resolve",
    "nameresolutionerror",
    "getaddrinfo failed",  # Windows
)


def has_dns_cause(exc: BaseException) -> bool:
    """True when `exc` (or anything in its __cause__/__context__ chain) is a DNS resolution failure."""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, socket.gaierror):
            return True
        if any(marker in str(cur).lower() for marker in _DNS_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False
