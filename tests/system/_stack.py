"""Helpers for the system tests: they talk to the RUNNING services over HTTP. Nothing here starts or stops anything.

The suite runs on the host against the published dev ports, or inside a container with the URLs overridden, e.g.

    docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T \
        -e IR_GATEWAY_URL=http://api_gateway:8000 -e IR_PREPROCESSING_URL=http://preprocessing:8001 \
        -e IR_INDEXING_URL=http://indexing:8002 -e IR_RETRIEVAL_URL=http://retrieval:8003 \
        -e IR_RANKING_URL=http://ranking:8004 retrieval python -m unittest discover -s tests/system -t .

A service that does not answer /health skips the suite with a message naming it, instead of failing.
"""
import json
import os
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

GATEWAY = os.environ.get("IR_GATEWAY_URL", "http://localhost:8000").rstrip("/")
PREPROCESSING = os.environ.get("IR_PREPROCESSING_URL", "http://localhost:8001").rstrip("/")
INDEXING = os.environ.get("IR_INDEXING_URL", "http://localhost:8002").rstrip("/")
RETRIEVAL = os.environ.get("IR_RETRIEVAL_URL", "http://localhost:8003").rstrip("/")
RANKING = os.environ.get("IR_RANKING_URL", "http://localhost:8004").rstrip("/")

SERVICES = {
    "api_gateway": GATEWAY,
    "preprocessing": PREPROCESSING,
    "indexing": INDEXING,
    "retrieval": RETRIEVAL,
    "ranking": RANKING,
}

# A dataset that already has bm25 + tfidf + embedding indices (override for another one that is indexed).
DATASET = os.environ.get("SYSTEM_TEST_DATASET", "cranfield")

# Cranfield's own query 1; its relevant document is 486 ("similarity laws for aerothermoelastic testing").
QUERY = "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft"
RELEVANT_DOC = "486"


class Response:
    """One HTTP answer: status, decoded body (dict/list when it is JSON) and elapsed milliseconds."""

    def __init__(self, status, body, raw, ms):
        self.status, self.body, self.raw, self.ms = status, body, raw, ms

    def json(self):
        return self.body

    def __repr__(self):
        return f"<Response {self.status} in {self.ms:.0f} ms {self.raw[:120]!r}>"


def call(service, path, body=None, method=None, params=None, timeout=180):
    """One request against `service` (a key of SERVICES). HTTP errors are returned, not raised."""
    url = SERVICES[service] + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            raw, status = answer.read().decode(), answer.status
    except urllib.error.HTTPError as error:  # 4xx/5xx are answers too
        raw, status = error.read().decode(), error.code
    elapsed = (time.perf_counter() - started) * 1000
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = raw
    return Response(status, parsed, raw, elapsed)


def health(service, timeout=5):
    """The /health status of a service, or 0 when it is unreachable."""
    try:
        with urllib.request.urlopen(SERVICES[service] + "/health", timeout=timeout) as answer:
            return answer.status
    except (urllib.error.URLError, OSError):
        return 0


def require_stack(services=None):
    """Skip the calling suite (from setUpModule) unless every needed service answers /health."""
    missing = [name for name in (services or SERVICES) if health(name) != 200]
    if missing:
        raise unittest.SkipTest("the system tests need the running stack; unreachable: " + ", ".join(missing))
