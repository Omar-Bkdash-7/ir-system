"""
Preprocessing core: config, pipeline (TextPreprocessor), resources, custom exceptions.

Public names are resolved on first access (PEP 562) instead of being imported
eagerly: eager imports here close a circular import with
`services/preprocessing/processors` (see `services/preprocessing/_lazy.py`).
Both the package-root path (e.g. `from services.preprocessing.core import
PreprocessingConfig`) and the submodule path (e.g. `from services.preprocessing.core.pipeline
import TextPreprocessor`) work.
"""

from services.preprocessing._lazy import install_lazy_exports

install_lazy_exports(globals(), __name__, {
    "PreprocessingConfig": "services.preprocessing.core.config",
    "TextPreprocessor": "services.preprocessing.core.pipeline",
    "preprocess": "services.preprocessing.core.pipeline",
    "get_default_preprocessor": "services.preprocessing.core.pipeline",
    "warmup_resources": "services.preprocessing.core.resources",
    "ResourceNotFoundError": "services.preprocessing.core.exceptions",
    "TokenizationError": "services.preprocessing.core.exceptions",
})

__all__ = [
    "PreprocessingConfig",
    "TextPreprocessor",
    "preprocess",
    "get_default_preprocessor",
    "warmup_resources",
    "ResourceNotFoundError",
    "TokenizationError",
]
