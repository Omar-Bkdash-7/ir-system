"""
CSS lives in `frontend/styles/`, not in Python string literals.

These checks keep the two halves in sync: every `inject_style("name")` must have
a stylesheet, and no stylesheet may be orphaned. They need neither Streamlit nor
a browser.
"""
import re
from pathlib import Path

import pytest

# `inject_style("name")` renders a style tag; `load_css("name")` embeds the CSS in
# markup that has to stay a single element (the result card).
STYLE_CALL = re.compile(r"(?:inject_style|load_css)\(\s*[\"']([a-z0-9_]+)[\"']")


def python_files(frontend_dir: Path):
    return [p for p in frontend_dir.rglob("*.py") if "__pycache__" not in p.parts]


def referenced_styles(frontend_dir: Path) -> set[str]:
    names = set()
    for path in python_files(frontend_dir):
        names.update(STYLE_CALL.findall(path.read_text(encoding="utf-8")))
    return names


def available_styles(frontend_dir: Path) -> set[str]:
    styles_dir = frontend_dir / "styles"
    return {p.stem for p in styles_dir.glob("*.css")}


def test_every_injected_style_exists(frontend_dir):
    missing = sorted(referenced_styles(frontend_dir) - available_styles(frontend_dir))
    assert missing == [], f"inject_style() names without a stylesheet: {missing}"


def test_no_stylesheet_is_orphaned(frontend_dir):
    """`theme` is injected by name too, so an orphan means a dead file."""
    unused = sorted(available_styles(frontend_dir) - referenced_styles(frontend_dir))
    assert unused == [], f"stylesheets never injected: {unused}"


def test_the_theme_and_the_shell_are_injected(frontend_dir):
    app_source = (frontend_dir / "app.py").read_text(encoding="utf-8")

    assert 'inject_style("theme"' in app_source
    assert 'inject_style("app_shell"' in app_source


def test_the_result_card_keeps_its_markup_and_rules_in_one_element(frontend_dir):
    source = (frontend_dir / "components" / "result_card.py").read_text(encoding="utf-8")

    assert "load_css('result_card_expand')" in source
    assert "inject_style(\"result_card_expand\")" not in source


# Only real string literals (`"<style>…"`), not prose in docstrings.
INLINE_STYLE = re.compile(r"f?[\"']<style>(.*?)</style>", re.S)


def test_no_css_rules_are_written_in_python(frontend_dir):
    """A `<style>` tag in Python may only wrap a placeholder, never hold rules."""
    offenders = []
    for path in python_files(frontend_dir):
        for body in INLINE_STYLE.findall(path.read_text(encoding="utf-8")):
            if re.fullmatch(r"\s*\{[^{}]*\}\s*", body) or not body.strip():
                continue
            offenders.append(f"{path.relative_to(frontend_dir).as_posix()}: {body.strip()[:40]}")

    assert offenders == [], f"CSS rules still inlined in Python: {offenders}"


@pytest.mark.parametrize("name", ["theme", "app_shell", "sidebar_nav", "tabs", "dataframe_tags"])
def test_core_stylesheets_contain_rules(frontend_dir, name):
    css = (frontend_dir / "styles" / f"{name}.css").read_text(encoding="utf-8")

    assert "{" in css and "}" in css
    assert len(css.strip()) > 40


def test_the_shared_stylesheets_are_reused_by_more_than_one_page(frontend_dir):
    """`tabs` and `dataframe_tags` were duplicated word for word before extraction."""
    for name in ("tabs", "dataframe_tags"):
        sites = [
            path.relative_to(frontend_dir).as_posix()
            for path in python_files(frontend_dir)
            if f'inject_style("{name}")' in path.read_text(encoding="utf-8")
        ]
        assert len(sites) >= 2, f"{name} is only injected by {sites}"
