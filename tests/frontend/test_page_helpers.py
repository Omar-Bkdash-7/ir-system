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


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers behind the catalogue filters and the guide copy
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Any", None),
        ("300", 300),
        ("2,000", 2000),
        ("10K", 10_000),
        ("1.5M", 1_500_000),
        ("2B", 2_000_000_000),
    ],
)
def test_parse_filter_reads_the_catalogue_size_box(raw, expected):
    from app_pages.dataset import _parse_filter

    assert _parse_filter(raw) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [(999, "999"), (1_500, "2K"), (24_000, "24K"), (2_500_000, "2.5M")],
)
def test_fmt_shortens_counts(count, expected):
    from app_pages.dataset import _fmt

    assert _fmt(count) == expected


def test_extrapolated_precision_curve_is_capped_and_ordered():
    """The P@K curve is an extrapolation: it must stay inside [0, 1]."""
    from app_pages.evaluation import _extrapolate_precision_at_k

    curve = _extrapolate_precision_at_k(0.9)
    assert curve[10] == 0.9
    assert all(0.0 <= value <= 1.0 for value in curve.values())
    assert curve[1] >= curve[10] >= curve[100]


def test_improvement_percent_handles_a_zero_baseline():
    from app_pages.evaluation import _improvement_percent

    assert _improvement_percent(0.0, 0.4) == 0
    assert _improvement_percent(0.4, 0.5) == pytest.approx(25.0)


def test_diff_html_marks_only_the_words_that_were_added():
    from app_pages.query_refinement import _diff_html

    html = _diff_html("information retrieval", "information retrival systems")
    assert '<span class="diff-unchanged">information</span>' in html
    assert '<span class="diff-added">systems</span>' in html
    assert '<span class="diff-unchanged">retrieval</span>' not in html


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        ("connection", "Connection refused"),
        ("timeout", "Request timed out after 30s"),
        ("server", "Service error (HTTP 500)"),
        ("client", "dataset not found"),
        (None, "Unknown error"),
    ],
)
def test_failure_hint_keeps_the_service_message_and_adds_a_hint(reason, detail):
    """A user must be able to tell a dead container from a rejected request."""
    from obs import failure_hint

    hint = failure_hint(StubResponse(ok=False, error=detail, reason=reason))
    assert detail in hint
    assert (len(hint) > len(detail)) == (reason in {"connection", "timeout", "server"})


def test_setup_logging_does_not_stack_handlers(monkeypatch, tmp_path):
    """Each line must be written once, however often the app is re-run."""
    import logging

    import obs

    monkeypatch.setattr(obs, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(obs, "_configured", False)

    obs.setup_logging()
    logger = logging.getLogger("frontend")
    handlers = len(logger.handlers)

    monkeypatch.setattr(obs, "_configured", False)
    obs.setup_logging()

    assert handlers == 2, "expected one stderr and one file handler"
    assert len(logger.handlers) == handlers


def test_load_css_serves_an_edit_without_a_process_restart(monkeypatch, tmp_path):
    """Editing a stylesheet must show up on the next rerun, not the next restart."""
    import os

    import ui_style

    monkeypatch.setattr(ui_style, "_STYLES_DIR", str(tmp_path))
    ui_style.clear_css_cache()

    path = tmp_path / "probe.css"
    path.write_text("a{color:red}", encoding="utf-8")
    assert ui_style.load_css("probe") == "a{color:red}"

    path.write_text("a{color:blue}", encoding="utf-8")
    os.utime(path, (0, 0))  # make sure the mtime really changes
    assert ui_style.load_css("probe") == "a{color:blue}"


def test_the_guide_copy_is_data_for_both_languages():
    from content import guide_ar, guide_en

    for module in (guide_en, guide_ar):
        kinds = [kind for kind, _ in module.BLOCKS]
        assert kinds.count("divider") == 2
        assert kinds.count("expander") >= 9

        for kind, payload in module.BLOCKS:
            if kind == "expander":
                assert payload["title"] and payload["body"].strip()
            elif kind in ("markdown", "html"):
                assert payload.strip()


def test_the_guide_page_renders_every_block_in_both_languages():
    script = """
import streamlit as st
import app_pages.guide as page

page.render()
"""
    app = AppTest.from_string(script, default_timeout=60)
    app.session_state["guide_lang"] = "en"
    app.run()
    assert list(app.exception) == []
    english_expanders = len(app.expander)

    app.session_state["guide_lang"] = "ar"
    app.run()
    assert list(app.exception) == []
    assert len(app.expander) == english_expanders


FAILING_SEARCH_SCRIPT = """
from types import SimpleNamespace

import api.gateway as gateway
import app_pages.search as page


def _fail(**kwargs):
    return SimpleNamespace(ok=False, data=None, error="Connection refused", reason="connection")


class _Box:
    def update(self, **kwargs):
        pass


gateway.search_hybrid_serial = _fail
page._search_hybrid_serial(
    "query",
    {"first_model": "bm25", "second_model": "embedding"},
    _Box(),
    0.0,
)
"""


def test_a_failed_hybrid_search_shows_a_hint_and_stores_nothing():
    app = AppTest.from_string(FAILING_SEARCH_SCRIPT, default_timeout=60)
    app.session_state["search_results"] = None
    app.run()

    assert list(app.exception) == []
    assert app.session_state["search_results"] is None
    messages = " ".join(str(err.value) for err in app.error)
    assert "Connection refused" in messages
    assert "not reachable" in messages


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
