"""HTTP client of the evaluation service — the implementation lives in `common.http_client`.

Two services now talk to their neighbours the same way (same retry policy, same
timeouts, same exception type), so the code lives once in `common/` and this
module keeps the import site the service and its tests use.
"""
from common.http_client import DownstreamError as UpstreamServiceError
from common.http_client import JsonClient

__all__ = ["JsonClient", "UpstreamServiceError"]
