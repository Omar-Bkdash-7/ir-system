"""CORS and rate limiting: the protections every service now shares.

`CORS_ORIGINS` used to be honoured by one service only, and `RATE_LIMIT` was
documented but never read by any code.
"""
import importlib.util
import os
import time
import unittest
from unittest import mock

HAVE_FASTAPI = all(importlib.util.find_spec(module) is not None for module in ("fastapi", "httpx"))


@unittest.skipUnless(HAVE_FASTAPI, "FastAPI test client (fastapi + httpx) is not installed")
class SecurityTest(unittest.TestCase):
    def client(self, *, limit=2, window=60.0, exempt=("/health",), origins=("http://ui.test",)):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from common.security import install_cors, install_rate_limit

        app = FastAPI()

        @app.get("/search")
        def search():
            return {"ok": True}

        @app.get("/health")
        def health():
            return {"status": "ok"}

        install_rate_limit(app, limit=limit, window=window, exempt=exempt)
        install_cors(app, origins=origins)
        return TestClient(app)

    # --------------------------------------------------------------------------------------------- throttling
    def test_requests_over_the_limit_are_refused_with_retry_after(self):
        client = self.client(limit=2)

        self.assertEqual([client.get("/search").status_code for _ in range(2)], [200, 200])
        refused = client.get("/search")

        self.assertEqual(refused.status_code, 429)
        self.assertEqual(refused.json()["error"], "RateLimitExceeded")
        self.assertGreaterEqual(int(refused.headers["retry-after"]), 1)

    def test_the_health_probe_is_never_throttled(self):
        client = self.client(limit=1)

        self.assertEqual([client.get("/health").status_code for _ in range(10)], [200] * 10)

    def test_a_new_window_starts_over(self):
        # The window has to be longer than two round trips through TestClient (~10 ms each).
        client = self.client(limit=1, window=0.5)

        self.assertEqual(client.get("/search").status_code, 200)
        self.assertEqual(client.get("/search").status_code, 429)
        time.sleep(0.6)
        self.assertEqual(client.get("/search").status_code, 200)

    def test_zero_limit_disables_the_limiter(self):
        client = self.client(limit=0)

        self.assertEqual([client.get("/search").status_code for _ in range(5)], [200] * 5)

    # --------------------------------------------------------------------------------------------- CORS
    def test_a_refused_request_still_carries_the_cors_headers(self):
        client = self.client(limit=1)
        client.get("/search", headers={"Origin": "http://ui.test"})

        refused = client.get("/search", headers={"Origin": "http://ui.test"})

        self.assertEqual(refused.status_code, 429)
        self.assertEqual(refused.headers.get("access-control-allow-origin"), "http://ui.test")

    def test_an_unlisted_origin_gets_no_cors_header(self):
        client = self.client(limit=10)

        response = client.get("/search", headers={"Origin": "http://elsewhere.test"})

        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_the_preflight_request_is_answered(self):
        client = self.client(limit=10)

        response = client.options(
            "/search",
            headers={"Origin": "http://ui.test", "Access-Control-Request-Method": "POST"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://ui.test")

    # --------------------------------------------------------------------------------------------- configuration
    def test_origins_come_from_the_environment(self):
        from common.security import cors_origins

        with mock.patch.dict(os.environ, {"CORS_ORIGINS": "http://a.test, http://b.test/ "}):
            self.assertEqual(cors_origins(), ["http://a.test", "http://b.test"])

        with mock.patch.dict(os.environ, {"CORS_ORIGINS": ""}):
            self.assertEqual(cors_origins(), [], "an empty list must mean 'no CORS', not 'everyone'")

    def test_the_limit_and_window_come_from_the_environment(self):
        from common.security import RateLimiter

        with mock.patch.dict(os.environ, {"RATE_LIMIT": "2", "RATE_LIMIT_WINDOW": "60"}):
            limiter = RateLimiter(2, 60.0)
            self.assertIsNone(limiter.hit("1.2.3.4"))
            self.assertIsNone(limiter.hit("1.2.3.4"))
            self.assertIsNotNone(limiter.hit("1.2.3.4"), "the third request is over a limit of two")
            self.assertIsNone(limiter.hit("5.6.7.8"), "a different client has its own window")


if __name__ == "__main__":
    unittest.main()
