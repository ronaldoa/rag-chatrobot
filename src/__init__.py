"""Source package exports."""
from .config import (
    GGUF_MODEL_PATH,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    DATA_PATH,
    VECTOR_STORE_PATH,
    validate_environment,
    print_config
)

__version__ = "1.0.0"
__all__ = [
    "GGUF_MODEL_PATH",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "DATA_PATH",
    "VECTOR_STORE_PATH",
    "validate_environment",
    "print_config"
]
