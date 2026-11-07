"""
Preprocessing processors: normalizer, tokenizer, filters, lemmatizers, stemmers.

Public names are resolved on first access (PEP 562) instead of being imported
eagerly: eager imports here close a circular import with `services.preprocessing.core`
(see `services/preprocessing/_lazy.py`). Both the package-root path
(e.g. `from services.preprocessing.processors import TokenFilter`) and the
submodule path (e.g. `from services.preprocessing.processors.filters import TokenFilter`) work.
"""

from services.preprocessing._lazy import install_lazy_exports

install_lazy_exports(globals(), __name__, {
    "SpacyTokenizer": "services.preprocessing.processors.tokenizer",
    "SimpleTokenizer": "services.preprocessing.processors.tokenizer",
    "TextNormalizer": "services.preprocessing.processors.normalizer",
    "TokenFilter": "services.preprocessing.processors.filters",
    "WordNetLemmatizerWrapper": "services.preprocessing.processors.lemmatizers",
    "PorterStemmerWrapper": "services.preprocessing.processors.stemmers",
})

__all__ = [
    "SpacyTokenizer",
    "SimpleTokenizer",
    "TextNormalizer",
    "TokenFilter",
    "WordNetLemmatizerWrapper",
    "PorterStemmerWrapper",
]
