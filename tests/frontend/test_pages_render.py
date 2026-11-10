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
