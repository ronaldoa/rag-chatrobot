"""Main entrypoint - Gradio + FastAPI hybrid app."""
import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pathlib import Path

from src.qa_service import qa_service
from src.config import DATA_PATH, VECTOR_STORE_PATH, GGUF_MODEL_PATH, validate_environment  # Load shared config
from api.chat import router as chat_router
from api.documents import router as docs_router
from ui.gradio_interface import create_gradio_interface

# ============ FastAPI app ============
app = FastAPI(
    title="Llama 3 Chatbot API",
    description="Local knowledge-base QA system - Gradio + FastAPI hybrid",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============ CORS ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Config validation ============
def validate_config_consistency():
    """
    Validate that critical paths are aligned across modules.

    Ensures all modules share the same configuration.
    """
    print("\n🔍 Validating configuration consistency...")

    issues = []

    # Check critical paths exist or can be created
    paths_to_check = {
        "Data directory": DATA_PATH,
        "Vector store directory": VECTOR_STORE_PATH,
        "Model file": GGUF_MODEL_PATH
    }

    for name, path in paths_to_check.items():
        p = Path(path)
        if name == "Model file":
            if not p.exists():
                issues.append(f"❌ {name} not found: {path}")
            else:
                print(f"  ✓ {name}: {path}")
        else:
            # Directories can be created automatically
            try:
                p.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ {name}: {path}")
            except Exception as e:
                issues.append(f"❌ Could not create {name}: {path} ({e})")

    if issues:
        print("\n⚠️  Configuration issues found:")
        for issue in issues:
            print(f"  {issue}")
        print()
    else:
        print("✅ Configuration looks good\n")

    return len(issues) == 0

# ============ Startup ============
@app.on_event("startup")
async def startup_event():
    """Initialize QA service on startup."""
    print("\n" + "="*50)
    print("🚀 Starting Llama 3 Chatbot service")
    print("="*50 + "\n")

    # Validate config
    if not validate_config_consistency():
        print("⚠️  Configuration issues detected, starting anyway\n")

    # Validate environment
    if not validate_environment():
        print("⚠️  Environment validation failed; some features may be unavailable\n")

    try:
        qa_service.initialize()
        print("✓ Service started\n")
        print("📍 Endpoints:")
        print("  • Gradio UI:   http://localhost:7860/")
        print("  • API docs:    http://localhost:7860/docs")
        print("  • Health:      http://localhost:7860/api/health")
        print("  • Path config: http://localhost:7860/api/config/paths")
        print("  • Doc config:  http://localhost:7860/api/documents/config")
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

# ============ API routes ============
app.include_router(chat_router)
app.include_router(docs_router)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Gradio UI."""
    return RedirectResponse(url="/gradio")

# ============ Mount Gradio ============
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# ============ Entrypoint ============
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info"
    )
