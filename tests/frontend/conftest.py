"""Make `frontend/` importable for the API-layer tests.

`frontend/` is a Streamlit app directory, not a package: `app.py` puts itself on
`sys.path` at runtime, so the tests do the same instead of adding `__init__.py`
files that would change how the app is launched.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"

if not FRONTEND.is_dir():
    raise RuntimeError(
        f"frontend/ not found at {FRONTEND}. The development containers mount it at "
        "/app/frontend (see docker-compose.dev.yml, service `clustering`)."
    )

if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))


@pytest.fixture
def frontend_dir() -> Path:
    return FRONTEND
