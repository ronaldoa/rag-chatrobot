"""Project configuration management."""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# ============ Paths ============
BASE_DIR = Path(__file__).parent.parent

# Load environment variables (prefer project .env)
load_dotenv(BASE_DIR / ".env")
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", str(BASE_DIR / "data"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "vector_store"))
LOG_PATH = os.getenv("LOG_PATH", str(BASE_DIR / "logs"))
MODELS_PATH = os.getenv("MODELS_PATH", str(BASE_DIR / "models"))

# ============ Model configuration ============
GGUF_MODEL_PATH = os.getenv(
    "GGUF_MODEL_PATH",
    str(BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf"),
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L6-v2"
)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# ============ Retrieval mode ============
_legacy_use_hybrid = os.getenv("USE_HYBRID", "True").lower() == "true"
RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", None)
if RETRIEVER_MODE:
    RETRIEVER_MODE = RETRIEVER_MODE.lower()
else:
    # Fallback to previous USE_HYBRID flag
    RETRIEVER_MODE = "hybrid" if _legacy_use_hybrid else "dense"
if RETRIEVER_MODE not in {"dense", "bm25", "hybrid"}:
    RETRIEVER_MODE = "hybrid"
USE_HYBRID = RETRIEVER_MODE == "hybrid"

# Optional weight for dense vs sparse when combining (dense_weight, sparse_weight)
HYBRID_WEIGHTS = os.getenv("HYBRID_WEIGHTS", "0.5,0.5")
# Multi-query expansion (paraphrase multiple queries then merge results)
MULTI_QUERY = os.getenv("MULTI_QUERY", "False").lower() == "true"
MULTI_QUERY_NUM = int(os.getenv("MULTI_QUERY_NUM", "3"))

# ============ LLM parameters ============
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", "16"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "35"))
N_BATCH = int(os.getenv("N_BATCH", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.15"))

# ============ Retrieval parameters ============
INITIAL_K = int(os.getenv("INITIAL_K", "20"))
FINAL_K = int(os.getenv("FINAL_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SEMANTIC_CHUNKING = os.getenv("SEMANTIC_CHUNKING", "False").lower() == "true"
SEMANTIC_BREAKPOINT_PERCENTILE = int(
    os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", "95")
)
SEMANTIC_MIN_CHUNK_SIZE = int(os.getenv("SEMANTIC_MIN_CHUNK_SIZE", "200"))

# ============ Server configuration ============
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
RELOAD = os.getenv("RELOAD", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
SHARE = os.getenv("SHARE", "False").lower() == "true"

# ============ Prompt template ============
# PROMPT_TEMPLATE = """<|start_header_id|>system<|end_header_id|>
#
# You are a precise QA assistant. Answer using ONLY the context below.
#
# Instructions:
# - Be concise: 1–3 sentences. No extra explanations or assumptions.
# - If context is insufficient, say exactly: "Based on the provided information, I cannot fully answer this question."
# - Do not invent details outside the context.
# - If bullet points help clarity, keep them brief.
#
# Context:
# {context}
#
# <|eot_id|><|start_header_id|>user<|end_header_id|>
#
# {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# """

# PROMPT_TEMPLATE = """<|start_header_id|>system<|end_header_id|>
#
# You are a QA assistant for a Retrieval-Augmented Generation (RAG) system
# about the book "Reminiscences of a Stock Operator".
#
# Instructions:
# 1. You MUST answer ONLY using the information in the Context.
# 2. If the Context does NOT contain enough information to answer,
#   you MUST say exactly: "Based on the provided information, I cannot fully answer this question."
# 3. Do NOT use any outside knowledge or assumptions.
# 4. Answer in 1–3 sentences.
#
# Context:
# {context}
#
# <|eot_id|><|start_header_id|>user<|end_header_id|>
#
# {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """

PROMPT_TEMPLATE = """<|start_header_id|>system<|end_header_id|>

You are a careful but practical QA assistant for a Retrieval-Augmented Generation (RAG) system
about the book "Reminiscences of a Stock Operator".

Guidelines:
1. Use the Context as your ONLY source of factual information. Do NOT invent facts that are not supported by the Context.
2. If the Context contains partial or indirect clues that are relevant to the Question, you MUST still answer as best you can,
   inferring simple implications from the given text.
3. ONLY IF the Context is clearly unrelated to the Question and contains no useful clues, answer exactly:
   "Based on the provided information, I cannot fully answer this question."
4. Answer concisely in 1–3 sentences. Do NOT mention the word "Context" or these instructions in your answer.

Context:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


PROMPT_TEMPLATE_PROD = """
You are an expert on "Reminiscences of a Stock Operator".

Instructions:
1. Always read and use the Context when it is relevant.
2. If the Context is incomplete but gives clues, combine it with your own knowledge of the book.
3. Only if nothing in the Context is related, rely fully on your knowledge of the book.
4. Be concise (1–3 sentences) and accurate.
5. If you are unsure, say so explicitly.

Context:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


# ============ Supported file extensions ============
SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".csv": "csv",
    ".html": "html",
    ".htm": "html",
}


# ============ Configuration helper ============
class Settings:
    """Configuration helper class."""

    @staticmethod
    def to_dict() -> Dict[str, Any]:
        """Export all configuration values as a dictionary."""
        return {
            # Paths
            "base_dir": str(BASE_DIR),
            "data_path": DATA_PATH,
            "vector_store_path": VECTOR_STORE_PATH,
            "log_path": LOG_PATH,
            "models_path": MODELS_PATH,
            # Models
            "gguf_model_path": GGUF_MODEL_PATH,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            # LLM parameters
            "n_ctx": N_CTX,
            "n_threads": N_THREADS,
            "n_gpu_layers": N_GPU_LAYERS,
            "n_batch": N_BATCH,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_p": TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
            # Retrieval
            "retriever_mode": RETRIEVER_MODE,
            "hybrid_weights": HYBRID_WEIGHTS,
            "multi_query": MULTI_QUERY,
            "multi_query_num": MULTI_QUERY_NUM,
            "initial_k": INITIAL_K,
            "final_k": FINAL_K,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "semantic_chunking": SEMANTIC_CHUNKING,
            "semantic_breakpoint_percentile": SEMANTIC_BREAKPOINT_PERCENTILE,
            "semantic_min_chunk_size": SEMANTIC_MIN_CHUNK_SIZE,
            # Server
            "server_host": SERVER_HOST,
            "server_port": SERVER_PORT,
            "reload": RELOAD,
            "log_level": LOG_LEVEL,
            "share": SHARE,
            # Supported formats
            "supported_extensions": list(SUPPORTED_EXTENSIONS.keys()),
        }

    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """Return basic model file information."""
        model_path = Path(GGUF_MODEL_PATH)
        return {
            "model_name": model_path.name if model_path.exists() else "Not Found",
            "model_size": f"{model_path.stat().st_size / (1024**3):.2f} GB"
            if model_path.exists()
            else "N/A",
            "model_exists": model_path.exists(),
        }


# ============ Environment validation ============
def validate_environment():
    """Validate required paths and model file."""
    errors = []

    # Ensure required directories exist
    for path_name, path_value in [
        ("DATA_PATH", DATA_PATH),
        ("MODELS_PATH", MODELS_PATH),
        ("LOG_PATH", LOG_PATH),
    ]:
        path = Path(path_value)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {path}")
            except Exception as e:
                errors.append(f"Could not create directory {path_name}: {e}")

    # Check model file presence
    if not Path(GGUF_MODEL_PATH).exists():
        errors.append(
            f"Model file missing: {GGUF_MODEL_PATH}\n"
            "Download from https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF"
        )

    if errors:
        print("\nEnvironment configuration issues:\n")
        for error in errors:
            print(f"- {error}")
        return False

    return True


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 60)
    print("Current configuration")
    print("=" * 60)
    print(f"Model path:      {GGUF_MODEL_PATH}")
    print(f"Data directory:  {DATA_PATH}")
    print(f"Vector store:    {VECTOR_STORE_PATH}")
    print(f"Context window:  {N_CTX}")
    print(f"CPU threads:     {N_THREADS}")
    print(f"GPU layers:      {N_GPU_LAYERS}")
    print(f"Chunk size:      {CHUNK_SIZE}")
    print(f"Chunk Oversize:  {CHUNK_OVERLAP}")
    if RETRIEVER_MODE == "dense":
        retrieval_mode = "Dense (FAISS + Reranker)"
    elif RETRIEVER_MODE == "bm25":
        retrieval_mode = "BM25 + Reranker"
    else:
        retrieval_mode = "Hybrid (BM25 + FAISS + Reranker)"
    weight_info = f" | weights (dense,bm25)={HYBRID_WEIGHTS}" if RETRIEVER_MODE == "hybrid" else ""
    print(f"Retrieval:       {retrieval_mode} (top-{INITIAL_K} → rerank top-{FINAL_K}){weight_info}")
    mq_status = f"On (x{MULTI_QUERY_NUM})" if MULTI_QUERY else "Off"
    print(f"Multi-query:     {mq_status}")
    if SEMANTIC_CHUNKING:
        semantic_kwargs = {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": SEMANTIC_BREAKPOINT_PERCENTILE,
        }
        print(
            "SemanticChunker: On "
            f"(percentile={SEMANTIC_BREAKPOINT_PERCENTILE}, "
            f"min_chunk_size={SEMANTIC_MIN_CHUNK_SIZE})"
        )
        print(f"  • init kwargs: {semantic_kwargs}")
    else:
        print("SemanticChunker: Off")
    print("=" * 60 + "\n")
    print("=" * 60 + "\n")
