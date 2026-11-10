"""The HTTP client is the single place where a service failure is classified."""
import pytest
import requests

from api import client


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retries must not slow the suite down."""
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)


def test_a_successful_call_returns_the_decoded_payload(monkeypatch):
    monkeypatch.setattr(client.requests, "request", lambda *a, **k: FakeResponse(200, {"ok": 1}))

    response = client.get("http://x/health")

    assert response.ok is True
    assert response.data == {"ok": 1}
    assert response.status_code == 200
    assert response.reason is None


def test_a_non_json_success_returns_the_raw_text(monkeypatch):
    monkeypatch.setattr(client.requests, "request", lambda *a, **k: FakeResponse(200, None, "plain"))

    assert client.get("http://x").data == "plain"


def test_a_client_error_is_not_retried_and_keeps_the_detail(monkeypatch):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append(args)
        return FakeResponse(404, {"detail": "dataset not found"})

    monkeypatch.setattr(client.requests, "request", fake_request)

    response = client.get("http://x/clusters/nope")

    assert response.ok is False
    assert response.reason == "client"
    assert response.status_code == 404
    assert response.error == "dataset not found"
    assert len(calls) == 1


def test_a_server_error_is_retried_then_reported(monkeypatch):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append(args)
        return FakeResponse(503, {"detail": "unavailable"})

    monkeypatch.setattr(client.requests, "request", fake_request)

    response = client.get("http://x/status", max_retries=3)

    assert response.ok is False
    assert response.reason == "server"
    assert response.status_code == 503
    assert len(calls) == 3


def test_a_server_error_that_recovers_on_retry_succeeds(monkeypatch):
    responses = [FakeResponse(500, {}), FakeResponse(200, {"fine": True})]
    monkeypatch.setattr(client.requests, "request", lambda *a, **k: responses.pop(0))

    response = client.get("http://x", max_retries=2)

    assert response.ok is True
    assert response.data == {"fine": True}


def test_a_connection_error_is_classified_as_connection(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(client.requests, "request", refuse)

    response = client.post("http://x", max_retries=2)

    assert response.reason == "connection"
    assert response.status_code is None
    assert "offline" in response.error


def test_a_timeout_is_classified_as_timeout(monkeypatch):
    def slow(*args, **kwargs):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(client.requests, "request", slow)

    response = client.post("http://x", timeout=7, max_retries=1)

    assert response.reason == "timeout"
    assert "7s" in response.error


def test_health_check_makes_a_single_attempt(monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(client.requests, "request", refuse)

    client.health_check("http://service:8000")

    assert len(calls) == 1


def test_the_response_is_truthy_only_when_it_succeeded(monkeypatch):
    monkeypatch.setattr(client.requests, "request", lambda *a, **k: FakeResponse(200, {}))
    assert bool(client.get("http://x")) is True

    monkeypatch.setattr(client.requests, "request", lambda *a, **k: FakeResponse(400, {"detail": "bad"}))
    assert bool(client.get("http://x")) is False
