"""
Embedding model helpers — BGE optimized version
"""

"""
Embedding model helpers — BGE optimized version
"""

import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from .config import EMBEDDING_MODEL


def get_embeddings(device: str = None):
    """
    Load a BGE embedding model with correct query instruction.
    """

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = "cpu"

    print(f"  • Embedding device: {device}")
    print(f"  • Embedding model: {EMBEDDING_MODEL}")

    # BGE 官方推荐的 query 指令
    QUERY_INSTRUCTION = "Represent this query for retrieving relevant documents:"

    model_kwargs = {
        "device": device,
        "trust_remote_code": True,
    }

    encode_kwargs = {
        "normalize_embeddings": True,
    }

    return HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction=QUERY_INSTRUCTION,
        # ⚠️ 这一版的 LangChain 不支持 text_instruction，所以不要传
    )
