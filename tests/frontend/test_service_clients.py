"""The service clients keep every gateway path, timeout and payload in one place."""
import pytest

from api import clustering, rag


@pytest.fixture
def captured(monkeypatch):
    """Replace the transport so the tests see exactly what would be sent."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(("GET", url, kwargs))
        return "response"

    def fake_post(url, **kwargs):
        calls.append(("POST", url, kwargs))
        return "response"

    def fake_delete(url, **kwargs):
        calls.append(("DELETE", url, kwargs))
        return "response"

    monkeypatch.setattr(clustering, "get", fake_get)
    monkeypatch.setattr(clustering, "post", fake_post)
    monkeypatch.setattr(clustering, "delete", fake_delete)
    monkeypatch.setattr(rag, "get", fake_get)
    monkeypatch.setattr(rag, "post", fake_post)
    return calls


def test_the_gateway_is_the_only_rag_entry_point(captured):
    rag.generate("q", "cranfield", 5, mode="local", temperature=0.1, max_tokens=64)

    method, url, kwargs = captured[0]
    assert (method, url) == ("POST", "http://api_gateway:8000/rag/generate")
    assert kwargs["json"] == {
        "query": "q",
        "dataset": "cranfield",
        "retrieval_model": "embedding",
        "top_k": 5,
        "generation_config": {
            "mode": "local",
            "temperature": 0.1,
            "max_tokens": 64,
            "use_context": True,
        },
    }
    assert kwargs["timeout"] == rag.GENERATE_TIMEOUT


def test_generation_waits_longer_than_the_gateway_allows(captured):
    rag.generate("q", "d", 1)
    assert captured[0][2]["timeout"] > 300


def test_rag_probes_use_short_timeouts(captured):
    rag.models()
    rag.local_model_status()
    assert [call[2]["timeout"] for call in captured] == [rag.PROBE_TIMEOUT, rag.PROBE_TIMEOUT]


def test_a_cached_clustering_run_does_not_wait_for_a_cold_run(captured):
    clustering.run("cranfield")

    method, url, kwargs = captured[0]
    assert (method, url) == ("POST", "http://clustering:8007/cluster")
    assert kwargs["timeout"] == clustering.CLUSTER_TIMEOUT
    assert kwargs["json"] == {
        "dataset": "cranfield",
        "force_recompute": False,
        "include_visualization": False,
    }


def test_a_forced_rebuild_uses_the_long_timeout_and_sends_the_parameters(captured):
    clustering.run(
        "cranfield",
        min_cluster_size=12,
        min_samples=4,
        metric="cosine",
        n_keywords=8,
        force_recompute=True,
        include_visualization=True,
        long_run=True,
    )

    kwargs = captured[0][2]
    assert kwargs["timeout"] == clustering.CLUSTER_LONG_TIMEOUT
    assert kwargs["json"] == {
        "dataset": "cranfield",
        "force_recompute": True,
        "include_visualization": True,
        "min_cluster_size": 12,
        "min_samples": 4,
        "metric": "cosine",
        "n_keywords": 8,
    }


def test_unset_parameters_are_left_out_so_the_service_defaults_apply(captured):
    clustering.run("cranfield", min_cluster_size=None, metric=None)

    assert "min_cluster_size" not in captured[0][2]["json"]
    assert "metric" not in captured[0][2]["json"]


def test_dataset_ids_with_slashes_are_passed_through_unchanged(captured):
    clustering.status("beir/quora")
    clustering.clusters("beir/quora")
    clustering.delete_run("beir/quora")

    assert [call[1] for call in captured] == [
        "http://clustering:8007/status/beir/quora",
        "http://clustering:8007/clusters/beir/quora",
        "http://clustering:8007/clusters/beir/quora",
    ]


def test_the_visualization_is_requested_explicitly(captured):
    clustering.clusters("cranfield", include_visualization=False)

    assert captured[0][2]["params"] == {"include_visualization": "false"}
