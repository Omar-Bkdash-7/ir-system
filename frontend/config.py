"""
Central configuration for the IR System frontend.

Every service URL, default parameter, and tunable constant lives here and is
read from environment variables where relevant. Nothing else in the codebase
should hard-code a URL or a default value -- import it from this module.
"""
import os


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Microservice base URLs (override via environment variables, e.g. in
# docker-compose.yml, when integrating into the larger project)
# ---------------------------------------------------------------------------
API_GATEWAY_URL = _env_str("API_GATEWAY_URL", "http://api_gateway:8000")
PREPROCESSING_URL = _env_str("PREPROCESSING_URL", "http://preprocessing:8001")
INDEXING_URL = _env_str("INDEXING_URL", "http://indexing:8002")
RETRIEVAL_URL = _env_str("RETRIEVAL_URL", "http://retrieval:8003")
RANKING_URL = _env_str("RANKING_URL", "http://ranking:8004")
QUERY_REFINEMENT_URL = _env_str("QUERY_REFINEMENT_URL", "http://query_refinement:8005")
EVALUATION_URL = _env_str("EVALUATION_URL", "http://evaluation:8006")
CLUSTERING_URL = _env_str("CLUSTERING_URL", "http://clustering:8007")

# Ordered registry used by the System Monitor page and the sidebar health grid.
SERVICES = {
    "API Gateway": {"url": API_GATEWAY_URL, "port": 8000, "health_path": "/health"},
    "Preprocessing": {"url": PREPROCESSING_URL, "port": 8001, "health_path": "/health"},
    "Indexing": {"url": INDEXING_URL, "port": 8002, "health_path": "/health"},
    "Retrieval": {"url": RETRIEVAL_URL, "port": 8003, "health_path": "/health"},
    "Ranking": {"url": RANKING_URL, "port": 8004, "health_path": "/health"},
    "Query Refinement": {"url": QUERY_REFINEMENT_URL, "port": 8005, "health_path": "/health"},
    "Evaluation": {"url": EVALUATION_URL, "port": 8006, "health_path": "/health"},
    "Clustering": {"url": CLUSTERING_URL, "port": 8007, "health_path": "/health"},
}

# ---------------------------------------------------------------------------
# HTTP client behavior (api/client.py)
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", 10)          # seconds, normal calls
BUILD_TIMEOUT = _env_int("BUILD_TIMEOUT", 1800)             # seconds, index builds
HEALTH_TIMEOUT = _env_int("HEALTH_TIMEOUT", 5)               # seconds, health pings
MAX_RETRIES = _env_int("MAX_RETRIES", 3)
RETRY_BACKOFF_BASE_SECONDS = 0.5

# ---------------------------------------------------------------------------
# Dataset (Dynamic - loaded from backend)
# ---------------------------------------------------------------------------
# Default values shown until dataset is loaded from backend
DATASET_NAME = "No dataset loaded"
DATASET_DOC_COUNT = 0
DATASET_QUERY_COUNT_APPROX = 0
DATASET_QREL_COUNT_APPROX = 0

# ---------------------------------------------------------------------------
# Retrieval defaults / bounds
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 50
MIN_TOP_K, MAX_TOP_K = 10, 500

DEFAULT_DISPLAY_K = 20
MIN_DISPLAY_K, MAX_DISPLAY_K = 5, 50

DEFAULT_BM25_K1 = 1.5
MIN_BM25_K1, MAX_BM25_K1 = 0.0, 3.0
DEFAULT_BM25_B = 0.75
MIN_BM25_B, MAX_BM25_B = 0.0, 1.0

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

TFIDF_DESCRIPTION = "sublinear_tf=True, norm=L2"

RETRIEVAL_MODELS = ["tfidf", "bm25", "embedding"]
RETRIEVAL_MODEL_LABELS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "embedding": "Embedding",
}
RETRIEVAL_MODEL_ICONS = {
    "tfidf": "📊",
    "bm25": "📈",
    "embedding": "🧠",
}

# Hybrid Serial (re-ranking pipeline)
HYBRID_SERIAL_DEFAULT_CANDIDATES = 1000
MIN_SERIAL_CANDIDATES, MAX_SERIAL_CANDIDATES = 100, 5000
HYBRID_SERIAL_DEFAULT_FINAL_K = 50
MIN_SERIAL_FINAL_K, MAX_SERIAL_FINAL_K = 10, 200

# Hybrid Parallel (fusion)
HYBRID_PARALLEL_DEFAULT_CANDIDATES = 1000
MIN_PARALLEL_CANDIDATES, MAX_PARALLEL_CANDIDATES = 100, 5000
HYBRID_PARALLEL_DEFAULT_FINAL_K = 50
MIN_PARALLEL_FINAL_K, MAX_PARALLEL_FINAL_K = 10, 200

RRF_DEFAULT_K = 60
MIN_RRF_K, MAX_RRF_K = 10, 100

FUSION_METHODS = ["rrf", "combsum", "combmnz", "weighted"]
FUSION_METHOD_LABELS = {
    "rrf": "RRF (Reciprocal Rank Fusion)",
    "combsum": "CombSUM",
    "combmnz": "CombMNZ",
    "weighted": "Weighted Average",
}
FUSION_METHOD_FORMULAS = {
    "rrf": "score = Σ 1 / (k + rank)",
    "combsum": "score = Σ normalized_score_i",
    "combmnz": "score = CombSUM × count_of_retrieving_models",
    "weighted": "score = Σ weight_i × normalized_score_i",
}

# ---------------------------------------------------------------------------
# Query refinement defaults
# ---------------------------------------------------------------------------
REFINEMENT_MAX_EDIT_DISTANCE_DEFAULT = 2
REFINEMENT_MAX_EDIT_DISTANCE_RANGE = (1, 3)

REFINEMENT_CONFIDENCE_THRESHOLD_DEFAULT = 0.7
REFINEMENT_CONFIDENCE_THRESHOLD_RANGE = (0.0, 1.0)

REFINEMENT_MAX_SYNONYMS_DEFAULT = 2
REFINEMENT_MAX_SYNONYMS_RANGE = (0, 5)

REFINEMENT_MAX_SYNSETS_DEFAULT = 2
REFINEMENT_MAX_SYNSETS_RANGE = (1, 5)

REFINEMENT_MIN_TERM_LENGTH_DEFAULT = 4
REFINEMENT_MIN_TERM_LENGTH_RANGE = (2, 8)

REFINEMENT_HISTORY_SIMILARITY_THRESHOLD_DEFAULT = 0.35
REFINEMENT_HISTORY_SIMILARITY_THRESHOLD_RANGE = (0.0, 1.0)

REFINEMENT_MAX_HISTORY_LENGTH_DEFAULT = 10
REFINEMENT_MAX_HISTORY_LENGTH_RANGE = (1, 20)

# ---------------------------------------------------------------------------
# Preprocessing defaults & presets
# ---------------------------------------------------------------------------
DEFAULT_PREPROCESSING_CONFIG = {
    "normalize_unicode": True,
    "lowercase": True,
    "remove_punctuation": True,
    "remove_stopwords": True,
    "min_token_length": 2,
    "max_token_length": None,
    "use_lemmatization": True,
    "use_stemming": False,
    "keep_numbers": False,
    "use_spacy_tokenizer": True,
    "custom_stopwords": [],
}

PREPROCESSING_PRESETS = {
    "Default (Academic)": dict(DEFAULT_PREPROCESSING_CONFIG),
    "Aggressive (Stemming)": {
        **DEFAULT_PREPROCESSING_CONFIG,
        "use_lemmatization": False,
        "use_stemming": True,
        "min_token_length": 3,
        "keep_numbers": False,
    },
    "Minimal (No Stopword Removal)": {
        **DEFAULT_PREPROCESSING_CONFIG,
        "remove_stopwords": False,
        "use_lemmatization": False,
        "use_stemming": False,
        "min_token_length": 1,
    },
}

MIN_TOKEN_LENGTH_RANGE = (1, 5)

# ---------------------------------------------------------------------------
# Index build time estimates (purely informational, shown in the UI)
# ---------------------------------------------------------------------------
BUILD_TIME_ESTIMATES = {
    "tfidf": "~1 min",
    "bm25": "~2 min",
    "embedding": "~15 min (500K docs)",
}

# ---------------------------------------------------------------------------
# Caching TTLs (seconds) -- see performance considerations in the spec
# ---------------------------------------------------------------------------
CACHE_TTL_HEALTH = 60           # Increased from 30 to reduce API calls
CACHE_TTL_INDEX_STATUS = 120    # Increased from 60 to reduce API calls
CACHE_TTL_DATASET_INFO = 600    # Increased from 300 to reduce API calls

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
ENVIRONMENT = _env_str("ENVIRONMENT", "development")
APP_TITLE = "IR System"
APP_ICON = "🔍"
EVALUATION_FALLBACK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "evaluation", "precomputed_metrics.json"
)
AUTO_REFRESH_INTERVAL_OPTIONS = [5, 10, 30]

# ---------------------------------------------------------------------------
# Clustering defaults
# ---------------------------------------------------------------------------
CLUSTERING_MIN_CLUSTER_SIZE_DEFAULT = 5
CLUSTERING_MIN_CLUSTER_SIZE_RANGE = (2, 20)

CLUSTERING_MIN_SAMPLES_DEFAULT = 3
CLUSTERING_MIN_SAMPLES_RANGE = (1, 10)

CLUSTERING_METRICS = ["euclidean", "cosine", "manhattan"]
