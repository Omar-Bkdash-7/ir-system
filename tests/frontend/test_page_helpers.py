"""
Focused checks for helpers extracted from the page modules.

These are cheap unit tests plus two `AppTest` runs that exercise the threaded
build path in `app_pages.indexing` — the part that a plain render test never
reaches.
"""
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from test_pages_render import StubResponse  # noqa: E402

# A build service answer for a single index.
BUILD_OK = {
    "status": "ready",
    "index_size_mb": 12.5,
    "vocab_size": 24_000,
}

BUILD_SCRIPT = """
import app_pages.indexing as page

page._build_single("tfidf", force=True)
"""


def test_index_label_covers_the_combined_run_and_a_single_index():
    import app_pages.indexing as page

    assert page._index_label("all") == ("All Indices", "~18 min (combined)")

    label, estimate = page._index_label("bm25")
    assert label and label != "bm25", "the label should be a human-readable name"
    assert estimate


def test_index_label_falls_back_for_an_unknown_index():
    import app_pages.indexing as page

    label, estimate = page._index_label("does-not-exist")
    assert label == "does-not-exist"
    assert estimate == "a few minutes"


def test_build_stages_exist_for_every_index_type_the_page_offers():
    import app_pages.indexing as page

    for index_type in ("all", "tfidf", "bm25", "embedding"):
        stages = page._BUILD_STAGES[index_type]
        assert stages[-1][0] == 100, f"{index_type} never reaches 100%"
        percentages = [pct for pct, _ in stages]
        assert percentages == sorted(percentages), f"{index_type} stages move backwards"


@pytest.mark.parametrize(
    ("raw", "expected_mb"),
    [
        ("1024 kB", 1.0),
        ("2048 kB", 2.0),
        ("12.5 MB", 12.5),
        ("2 GB", 2048.0),
        ("1 TB", 1024 * 1024.0),
        ("512 B", 512 / (1024 * 1024)),
        ("—", 0.0),
        ("", 0.0),
        (None, 0.0),
    ],
)
def test_parse_size_to_mb_converts_postgres_size_strings(raw, expected_mb):
    from app_pages.dataset import _parse_size_to_mb

    assert _parse_size_to_mb(raw) == pytest.approx(expected_mb)


def test_a_successful_build_is_logged_and_does_not_raise(monkeypatch, frontend_dir):
    import api.indexing as indexing_api

    monkeypatch.setattr(indexing_api, "build_index", lambda *a, **k: StubResponse(data=BUILD_OK))
    monkeypatch.setattr(indexing_api, "clear_index_status_cache", lambda: None)

    app = AppTest.from_string(BUILD_SCRIPT, default_timeout=180)
    app.run()

    assert list(app.exception) == []
    log = " ".join(app.session_state["build_log"])
    assert "[OK] TF-IDF" in log, log
    assert "size=12.5MB" in log


def test_the_shared_dataset_selector_preselects_the_remembered_dataset():
    """A dataset picked on one page must come back on the next one."""
    script = """
import streamlit as st
from constants import SS_SELECTED_DATASET
from ui_state import dataset_selector

st.session_state[SS_SELECTED_DATASET] = "vaswani"
chosen = dataset_selector(["cranfield", "vaswani"], key="probe_dataset")
st.session_state["probe_result"] = chosen
"""
    app = AppTest.from_string(script, default_timeout=60)
    app.run()

    assert list(app.exception) == []
    assert app.session_state["probe_dataset"] == "vaswani"
    assert app.session_state["probe_result"] == "vaswani"


def test_the_shared_dataset_selector_falls_back_to_the_placeholder():
    """A remembered dataset that is not offered must not raise."""
    script = """
import streamlit as st
from constants import SS_SELECTED_DATASET
from ui_state import dataset_selector

st.session_state[SS_SELECTED_DATASET] = "gone"
chosen = dataset_selector(["cranfield"], key="probe_dataset")
st.session_state["probe_result"] = chosen
"""
    app = AppTest.from_string(script, default_timeout=60)
    app.run()

    assert list(app.exception) == []
    assert app.session_state["probe_result"] is None


def test_coming_back_to_a_finished_build_reports_it(monkeypatch, frontend_dir):
    """A build started on an earlier run must not be lost on navigation."""
    import api.indexing as indexing_api

    monkeypatch.setattr(indexing_api, "clear_index_status_cache", lambda: None)

    job = {
        "index_type": "tfidf",
        "label": "TF-IDF",
        "estimate": "a few minutes",
        "dataset_in_db": True,
        "stages": [(100, "Complete!")],
        "stage": 0,
        "started_at": 0.0,
        "done": True,
        "reported": False,
        "resp": StubResponse(data=BUILD_OK),
    }
    script = """
import app_pages.indexing as page

page._render_active_build()
"""

    app = AppTest.from_string(script, default_timeout=60)
    app.session_state["build_job"] = job
    app.run()

    assert list(app.exception) == []
    assert app.session_state["build_job"]["reported"] is True
    assert "[OK] TF-IDF" in " ".join(app.session_state["build_log"])


def test_a_failed_build_renders_the_error_panel(monkeypatch, frontend_dir):
    import api.indexing as indexing_api

    monkeypatch.setattr(
        indexing_api,
        "build_index",
        lambda *a, **k: StubResponse(ok=False, data=None, error="boom", status_code=500),
    )
    monkeypatch.setattr(indexing_api, "clear_index_status_cache", lambda: None)

    app = AppTest.from_string(BUILD_SCRIPT, default_timeout=180)
    app.run()

    assert list(app.exception) == []
    log = " ".join(app.session_state["build_log"])
    assert "[ERR]" in log, log
    assert "boom" in log
