# config.py

# ---------------------------
# Embeddings
# ---------------------------

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------
# Vector search
# ---------------------------

VECTOR_TOP_K = 5

# ---------------------------
# Web ingestion limits
# ---------------------------

MIN_RAW_CHARS = 1200
MAX_RAW_CHARS = 8000

MAX_SUMMARY_TOKENS = 1000

# ---------------------------
# Modes
# ---------------------------

MODES = {
    "quick": {
        "subqueries": 2,
    },
    "standard": {
        "subqueries": 4,
    },
}
