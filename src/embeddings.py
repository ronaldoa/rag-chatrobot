"""Embedding model helpers."""
from langchain.embeddings import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL
import torch

def get_embeddings(device: str = None):
    """
    Load the embedding model.

    Args:
        device: Target device ('cpu', 'cuda', 'mps', or None for auto-detect)

    Returns:
        HuggingFaceEmbeddings instance
    """
    if device is None:
        # Auto-detect device
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon
        else:
            device = 'cpu'

    print(f"  • Embedding device: {device}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            'device': device,
            'trust_remote_code': True
        },
        encode_kwargs={
            'normalize_embeddings': True,  # Normalize vectors
            'batch_size': 32  # Batch size
        }
    )
