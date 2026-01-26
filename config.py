# config.py

# ---------------------------
# Vector search
# ---------------------------

VECTOR_TOP_K = 10
CROSS_TOP_K = 5

# ---------------------------
# Web ingestion limits
# ---------------------------

MIN_RAW_CHARS = 1200
MAX_RAW_CHARS = 8000

MAX_SUMMARY_TOKENS = 1000

# ---------------------------
# Research modes
# ---------------------------

MODES = {
    "quick": {
        "iterations": 1,              # only initial discovery
        "queries_per_iteration": 2,
    },
    "standard": {
        "iterations": 2,              # +1 coverage refinement
        "queries_per_iteration": 2,
    },
    "deep": {
        "iterations": 3,              # +2 coverage refinements
        "queries_per_iteration": 2,
    },
}
