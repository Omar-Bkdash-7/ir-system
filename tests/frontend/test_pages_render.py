"""
Render the real Streamlit pages with stubbed service responses.

`AppTest` executes `frontend/app.py` exactly as Streamlit does, so these checks
catch a broken refactor (missing argument, unreachable section) without a
browser. They need Streamlit, which only the frontend image installs, and are
skipped elsewhere.
"""
from dataclasses import dataclass
from typing import Any, Optional

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


@dataclass
class StubResponse:
    """Mimics `api.client.APIResponse` for the pages under test."""

    ok: bool = True
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = 200
    elapsed_ms: float = 0.0
    reason: Optional[str] = None


SAMPLE_METRICS = {
    "TF-IDF": {"MAP": 0.3120, "Recall": 0.5210, "P@10": 0.2410, "nDCG@10": 0.2905},
    "BM25": {"MAP": 0.3380, "Recall": 0.5380, "P@10": 0.2570, "nDCG@10": 0.3120},
    "Embedding": {"MAP": 0.4010, "Recall": 0.6120, "P@10": 0.2990, "nDCG@10": 0.3710},
}

HISTORY = {
    "history": [
        {
            "id": 1,
            "model": "tfidf",
            "dataset": "cranfield",
            "map": 0.312,
            "ndcg": 0.29,
            "recall": 0.52,
            "precision": 0.24,
            "num_queries": 225,
            "metadata": {"enable_refinement": False},
        },
        {
            "id": 2,
            "model": "bm25",
            "dataset": "cranfield",
            "map": 0.338,
            "ndcg": 0.31,
            "recall": 0.54,
            "precision": 0.26,
            "num_queries": 225,
            "metadata": {"enable_refinement": True},
        },
    ]
}


def run_page(frontend_dir, page: str) -> AppTest:
    app = AppTest.from_file(str(frontend_dir / "app.py"), default_timeout=180)
    app.session_state["selected_page"] = page
    app.run()
    return app


def test_the_evaluation_dashboard_renders_every_section(monkeypatch, frontend_dir):
    import api.evaluation as eval_api

    monkeypatch.setattr(eval_api, "get_metrics", lambda: StubResponse(data=SAMPLE_METRICS))
    monkeypatch.setattr(eval_api, "get_history", lambda *a, **k: StubResponse(data=HISTORY))
    monkeypatch.setattr(
        eval_api, "get_per_query_analysis", lambda *a, **k: StubResponse(data={"queries": []})
    )

    app = run_page(frontend_dir, "evaluation")

    assert list(app.exception) == []
    headings = " ".join(str(m.value) for m in app.markdown)
    for section in ("Model Performance Overview", "Visual Analysis", "Quick Insights"):
        assert section in headings, f"missing section: {section}"
    assert len(app.tabs) >= 1, "the visual-analysis tabs did not render"
    assert len(app.dataframe) >= 1, "the history table did not render"


def test_the_evaluation_page_degrades_gracefully_without_live_data(monkeypatch, frontend_dir):
    import api.evaluation as eval_api

    monkeypatch.setattr(eval_api, "get_metrics", lambda: StubResponse(ok=False, data=None, error="offline"))
    monkeypatch.setattr(eval_api, "get_history", lambda *a, **k: StubResponse(ok=False, data=None))

    app = run_page(frontend_dir, "evaluation")

    assert list(app.exception) == []
    # Either the bundled pre-computed snapshot renders, or the page says why it cannot.
    assert app.dataframe or app.warning


READY = {"status": "ready", "index_size_mb": 12.5, "num_documents": 1398, "build_time": "2.1s"}


def test_the_indexing_page_renders_with_a_selected_dataset(monkeypatch, frontend_dir):
    """The page must render its actions, index cards and log without a dataset switch."""
    import api.gateway as gateway
    import api.indexing as indexing_api
    import app_pages.indexing as page

    monkeypatch.setattr(page, "_get_downloaded_datasets", lambda: ["cranfield"])
    monkeypatch.setattr(
        gateway,
        "get_dataset_info_cached",
        lambda: {"dataset_name": "cranfield", "doc_count": 1398, "query_count": 225, "qrel_count": 1837},
    )
    monkeypatch.setattr(
        indexing_api,
        "get_dataset_metadata",
        lambda dataset: {
            "dataset_name": dataset,
            "doc_count": 1398,
            "query_count": 225,
            "qrel_count": 1837,
        },
    )
    monkeypatch.setattr(indexing_api, "index_status", lambda *a, **k: StubResponse(data=READY))

    app = AppTest.from_file(str(frontend_dir / "app.py"), default_timeout=180)
    app.session_state["selected_page"] = "indexing"
    app.session_state["dataset_selector"] = "cranfield"
    app.run()

    assert list(app.exception) == []
    headings = " ".join(str(m.value) for m in app.markdown)
    assert "Index Status" in headings
    metrics = {str(m.label): m.value for m in app.metric}
    assert metrics.get("Docs") == "1,398"
    buttons = " ".join(str(b.label) for b in app.button)
    assert "Build All Indices" in buttons


def _clustering_page(monkeypatch, frontend_dir, datasets):
    """Render the clustering page with one dataset in the database."""
    import api.clustering as clustering_api
    import api.indexing as indexing_api
    import app_pages.clustering as page

    monkeypatch.setattr(page, "_get_datasets_with_embeddings", lambda: datasets)
    monkeypatch.setattr(
        indexing_api,
        "get_db_info",
        lambda: StubResponse(data={"datasets": [{"dataset_id": "cranfield", "doc_count": 1398}]}),
    )
    monkeypatch.setattr(clustering_api, "status", lambda dataset: StubResponse(data={"status": "not_clustered"}))

    app = AppTest.from_file(str(frontend_dir / "app.py"), default_timeout=180)
    app.session_state["selected_page"] = "clustering"
    app.session_state["selected_dataset"] = "cranfield"
    app.run()
    return app


def test_the_clustering_page_offers_the_run_action(monkeypatch, frontend_dir):
    app = _clustering_page(monkeypatch, frontend_dir, ["cranfield"])

    assert list(app.exception) == []
    buttons = " ".join(str(b.label) for b in app.button)
    assert "Run Clustering" in buttons


def test_the_clustering_page_reaches_the_service_when_run_is_pressed(monkeypatch, frontend_dir):
    """Pressing Run must reach the service; a leftover guard once made this a NameError."""
    import api.clustering as clustering_api

    calls = []

    def fake_run(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return StubResponse(ok=False, data=None, error="service offline")

    monkeypatch.setattr(clustering_api, "run", fake_run)
    app = _clustering_page(monkeypatch, frontend_dir, ["cranfield"])

    next(b for b in app.button if "Run Clustering" in str(b.label)).click().run()

    assert list(app.exception) == []
    assert calls and calls[0][0] == "cranfield"
    assert any("Clustering failed" in str(e.value) for e in app.error)


def test_the_clustering_page_explains_why_it_cannot_run(monkeypatch, frontend_dir):
    """The help screen used to read a variable that only existed inside render()."""
    app = _clustering_page(monkeypatch, frontend_dir, [])

    assert list(app.exception) == []
    assert any("No datasets with embeddings" in str(w.value) for w in app.warning)
