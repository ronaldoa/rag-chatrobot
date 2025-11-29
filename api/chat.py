"""Chat API endpoints."""
from fastapi import APIRouter, HTTPException
from .models import ChatRequest, ChatResponse, HealthResponse
from src.qa_service import qa_service
from src.config import DATA_PATH, VECTOR_STORE_PATH, GGUF_MODEL_PATH  # Shared paths
from pathlib import Path

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint.

    - **message**: User question
    - **include_sources**: Whether to include sources
    """
    try:
        result = qa_service.ask(request.message)

        response = ChatResponse(
            answer=result["answer"],
            sources=result["sources"] if request.include_sources else None
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    health_status = qa_service.health_check()
    return HealthResponse(
        status=health_status["status"],
        model_loaded=health_status["model_loaded"]
    )

@router.get("/config/paths")  # Path configuration inspection endpoint
async def get_config_paths():
    """
    Return configured paths to help debug configuration mismatches.
    """
    return {
        "data_path": {
            "configured": DATA_PATH,
            "resolved": str(Path(DATA_PATH).resolve()),
            "exists": Path(DATA_PATH).exists()
        },
        "vector_store_path": {
            "configured": VECTOR_STORE_PATH,
            "resolved": str(Path(VECTOR_STORE_PATH).resolve()),
            "exists": Path(VECTOR_STORE_PATH).exists()
        },
        "model_path": {
            "configured": GGUF_MODEL_PATH,
            "resolved": str(Path(GGUF_MODEL_PATH).resolve()),
            "exists": Path(GGUF_MODEL_PATH).exists()
        }
    }
