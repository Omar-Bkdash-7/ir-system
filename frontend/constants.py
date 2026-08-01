"""
Centralized `st.session_state` keys.

Never write a session-state string literal anywhere else in the codebase --
import the constant from this module instead. `DEFAULTS` is consumed once by
` ` on startup to seed every key with `st.session_state.setdefault`.
"""

# --- Navigation --------------------------------------------------------
SS_SELECTED_PAGE = "selected_page"

# --- Search page: query + execution state ------------------------------
SS_QUERY_HISTORY = "query_history"
SS_CURRENT_QUERY = "current_query"
SS_SEARCH_RESULTS = "search_results"
SS_LAST_SEARCH_TIME = "last_search_time"
SS_SEARCH_COUNT = "search_count"
SS_RETRIEVAL_MODE = "retrieval_mode"
SS_EXPANDED_DOCS = "expanded_docs"

# --- Search page: per-mode configuration --------------------------------
SS_INDIVIDUAL_MODEL_CONFIG = "individual_model_config"
SS_HYBRID_SERIAL_CONFIG = "hybrid_serial_config"
SS_HYBRID_PARALLEL_CONFIG = "hybrid_parallel_config"

# --- Preprocessing (Search panel + Dataset & Preprocessing page) -------
SS_PREPROCESSING_CONFIG = "preprocessing_config"
SS_PREPROCESSING_PRESET = "preprocessing_preset"
SS_SHOW_PREPROCESSING_PREVIEW = "show_preprocessing_preview"
SS_LAST_PREPROCESS_PREVIEW = "last_preprocess_preview"
SS_PREPROCESSING_LAB_INPUT = "preprocessing_lab_input"
SS_PREPROCESSING_LAB_RESULT = "preprocessing_lab_result"

# --- Query refinement toggle inline on the Search page ------------------
SS_REFINEMENT_ENABLED = "refinement_enabled"
SS_REFINEMENT_RESULT = "refinement_result"
SS_REFINEMENT_HISTORY = "refinement_history"

# --- Query Refinement Lab (dedicated page) ------------------------------
SS_REFINEMENT_LAB_CONFIG = "refinement_lab_config"
SS_REFINEMENT_LAB_RESULT = "refinement_lab_result"
SS_REFINEMENT_LAB_HISTORY_INPUT = "refinement_lab_history_input"
SS_REFINEMENT_LAB_QUERY_INPUT = "refinement_lab_query_input"

# --- Index management -----------------------------------------------------
SS_INDEX_STATUS = "index_status_cache"
SS_BUILD_LOG = "build_log"
SS_BUILD_IN_PROGRESS = "build_in_progress"   # False | "tfidf" | "bm25" | "embedding" | "all"
SS_FORCE_REBUILD = "force_rebuild"

# --- System monitor ---------------------------------------------------
SS_SERVICE_HEALTH = "service_health_cache"
SS_AUTO_REFRESH = "auto_refresh"
SS_AUTO_REFRESH_INTERVAL = "auto_refresh_interval"
SS_LAST_REFRESH_TIME = "last_refresh_time"
SS_TEST_PING_RESULTS = "test_ping_results"
SS_RETRIEVAL_DEEP_STATUS = "retrieval_deep_status_cache"

# --- Dataset ------------------------------------------------------------
SS_DATASET_INFO = "dataset_info_cache"
SS_DATASET_VALIDATION_RESULT = "dataset_validation_result"
SS_VALIDATION_IN_PROGRESS = "dataset_validation_in_progress"

# --- Misc -----------------------------------------------------------------
SS_SESSION_ID = "session_id"

# ---------------------------------------------------------------------------
# Static defaults, seeded in app.py at startup. Keys whose default depends on
# `config.py` (e.g. SS_PREPROCESSING_CONFIG) are seeded to None here and
# populated lazily on first use so this module never imports `config`.
# ---------------------------------------------------------------------------
DEFAULTS = {
    SS_SELECTED_PAGE: "search",

    SS_QUERY_HISTORY: [],
    SS_CURRENT_QUERY: "",
    SS_SEARCH_RESULTS: None,
    SS_LAST_SEARCH_TIME: None,
    SS_SEARCH_COUNT: 0,
    SS_RETRIEVAL_MODE: "individual",
    SS_EXPANDED_DOCS: set(),

    SS_INDIVIDUAL_MODEL_CONFIG: None,
    SS_HYBRID_SERIAL_CONFIG: None,
    SS_HYBRID_PARALLEL_CONFIG: None,

    SS_PREPROCESSING_CONFIG: None,
    SS_PREPROCESSING_PRESET: "Default (Academic)",
    SS_SHOW_PREPROCESSING_PREVIEW: False,
    SS_LAST_PREPROCESS_PREVIEW: None,
    SS_PREPROCESSING_LAB_INPUT: "What is the best way to learn machine learning?",
    SS_PREPROCESSING_LAB_RESULT: None,

    SS_REFINEMENT_ENABLED: False,
    SS_REFINEMENT_RESULT: None,
    SS_REFINEMENT_HISTORY: [],

    SS_REFINEMENT_LAB_CONFIG: None,
    SS_REFINEMENT_LAB_RESULT: None,
    SS_REFINEMENT_LAB_HISTORY_INPUT: "",
    SS_REFINEMENT_LAB_QUERY_INPUT: "",

    SS_INDEX_STATUS: {},
    SS_BUILD_LOG: [],
    SS_BUILD_IN_PROGRESS: False,
    SS_FORCE_REBUILD: False,

    SS_SERVICE_HEALTH: {},
    SS_AUTO_REFRESH: False,
    SS_AUTO_REFRESH_INTERVAL: 10,
    SS_LAST_REFRESH_TIME: None,
    SS_TEST_PING_RESULTS: {},
    SS_RETRIEVAL_DEEP_STATUS: None,

    SS_DATASET_INFO: None,
    SS_DATASET_VALIDATION_RESULT: None,
    SS_VALIDATION_IN_PROGRESS: False,
}