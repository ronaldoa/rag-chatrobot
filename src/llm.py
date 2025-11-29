"""LLM model helpers."""
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pathlib import Path

from .config import (
    GGUF_MODEL_PATH,
    N_CTX,
    N_THREADS,
    N_GPU_LAYERS,
    N_BATCH,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_P,
    REPEAT_PENALTY,
)


def _coerce_params(overrides):
    """Coerce UI-provided override values to correct types."""
    if not overrides:
        return {}

    def _maybe_int(value):
        try:
            return int(float(value))
        except Exception:
            return value

    def _maybe_float(value):
        try:
            return float(value)
        except Exception:
            return value

    coerced = {}
    for key, val in overrides.items():
        if val is None:
            continue
        if key in {"n_ctx", "n_threads", "n_gpu_layers", "n_batch"}:
            coerced[key] = _maybe_int(val)
        elif key in {"temperature", "top_p", "repeat_penalty"}:
            coerced[key] = _maybe_float(val)
        elif key == "max_tokens":
            coerced[key] = _maybe_int(val)
        elif key == "model_path":
            coerced[key] = str(val)
        else:
            coerced[key] = val
    return coerced


def get_llm(streaming: bool = False, verbose: bool = False, overrides: dict | None = None):
    """
    Load the LLM.

    Args:
        streaming: Enable streaming callbacks.
        verbose: Enable verbose logging.
        overrides: Optional dict of runtime overrides for LLM params.

    Returns:
        LlamaCpp instance
    """
    overrides = _coerce_params(overrides)

    params = {
        "model_path": overrides.get("model_path", GGUF_MODEL_PATH),
        "n_ctx": overrides.get("n_ctx", N_CTX),
        "n_threads": overrides.get("n_threads", N_THREADS),
        "n_gpu_layers": overrides.get("n_gpu_layers", N_GPU_LAYERS),
        "n_batch": overrides.get("n_batch", N_BATCH),
        "temperature": overrides.get("temperature", TEMPERATURE),
        "max_tokens": overrides.get("max_tokens", MAX_TOKENS),
        "top_p": overrides.get("top_p", TOP_P),
        "repeat_penalty": overrides.get("repeat_penalty", REPEAT_PENALTY),
    }

    # Verify model file
    model_path = Path(params["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {params['model_path']}\n"
            "Download the model and place it at the configured path."
        )

    # Callbacks
    callback_manager = None
    if streaming:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    print(f"  • Model file: {model_path.name}")
    print(f"  • Context window: {params['n_ctx']} tokens")
    print(f"  • CPU threads: {params['n_threads']}")
    gpu_mode = "(CPU only)" if params["n_gpu_layers"] == 0 else "(GPU accelerated)"
    print(f"  • GPU layers: {params['n_gpu_layers']} {gpu_mode}")

    return LlamaCpp(
        model_path=str(model_path),
        n_ctx=params["n_ctx"],
        n_threads=params["n_threads"],
        n_gpu_layers=params["n_gpu_layers"],
        n_batch=params["n_batch"],
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        top_p=params["top_p"],
        repeat_penalty=params["repeat_penalty"],
        callback_manager=callback_manager,
        verbose=verbose,
        # Performance tweaks
        use_mlock=True,  # Lock memory to avoid swapping
        use_mmap=True,   # Memory map for faster loading
        # f16_kv=True,   # Use FP16 KV cache (saves memory)
    )
