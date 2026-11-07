"""Preprocessing Service Package.

NLP text preprocessing for the IR system: normalization, tokenization,
stopword/length filtering, and stemming/lemmatization.

Public entry points are lazily re-exported from `services.preprocessing.preprocessing_service`
(PEP 562, see `services/preprocessing/_lazy.py`):
    from services.preprocessing import preprocess, preprocess_batch, PreprocessingConfig
"""

from services.preprocessing._lazy import install_lazy_exports

install_lazy_exports(globals(), __name__, {
    "preprocess": "services.preprocessing.preprocessing_service",
    "preprocess_batch": "services.preprocessing.preprocessing_service",
    "PreprocessingConfig": "services.preprocessing.preprocessing_service",
    "TextPreprocessor": "services.preprocessing.preprocessing_service",
    "get_default_preprocessor": "services.preprocessing.preprocessing_service",
    "get_preprocessor": "services.preprocessing.preprocessing_service",
    "warmup_resources": "services.preprocessing.preprocessing_service",
    "ResourceNotFoundError": "services.preprocessing.preprocessing_service",
    "TokenizationError": "services.preprocessing.preprocessing_service",
})

__all__ = [
    "preprocess",
    "preprocess_batch",
    "PreprocessingConfig",
    "TextPreprocessor",
    "get_default_preprocessor",
    "get_preprocessor",
    "warmup_resources",
    "ResourceNotFoundError",
    "TokenizationError",
]
