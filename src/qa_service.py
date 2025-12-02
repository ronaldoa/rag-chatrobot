"""QA business logic shared by API and Gradio."""
import os
from typing import Dict, List, Optional
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .config import (
    VECTOR_STORE_PATH,
    PROMPT_TEMPLATE,
    Settings,
    GGUF_MODEL_PATH,
    N_CTX,
    N_THREADS,
    N_GPU_LAYERS,
    N_BATCH,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_P,
    REPEAT_PENALTY,
    USE_HYBRID,
)  # Shared configuration helper
from .embeddings import get_embeddings
from .llm import get_llm
from .retriever import get_retriever

class QAService:
    """Question answering service."""

    def __init__(self):
        self.qa_chain = None
        self._initialized = False
        self._retriever = None
        self._llm_overrides = {}
        self._vectorstore = None
        self._embeddings = None
        self._llm = None

    def _build_chain(self):
        """Build RetrievalQA chain with current retriever/LLM."""
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=self._retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def _load_llm(self, overrides: dict | None = None):
        """Load LLM with optional runtime overrides."""
        if overrides:
            # Persist overrides so re-builds retain latest values
            self._llm_overrides.update(overrides)
        self._llm = get_llm(overrides=self._llm_overrides)

    def _format_history(self, history: Optional[List[Dict]], max_messages: int = 8, max_chars: int = 4000) -> str:
        """
        Serialize recent chat history into a plain-text block for prompting.
        Limits message count and total length to avoid oversized prompts.
        """
        if not history:
            return ""

        recent = history[-max_messages:]
        lines = []
        for msg in recent:
            role = msg.get("role", "user")
            role_label = "User" if role == "user" else "Assistant"
            content = msg.get("content", "")
            lines.append(f"{role_label}: {content}")

        joined = "\n".join(lines)
        return joined[:max_chars]

    def initialize(self):
        """Initialize the service."""
        if self._initialized:
            return

        print("🔧 Initializing QA service...")

        # Ensure vector store exists
        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {VECTOR_STORE_PATH}!\n"
                "Please run: python ingest.py"
            )

        # Load components
        print("  ├─ Loading embedding model...")
        self._embeddings = get_embeddings()

        print("  ├─ Loading vector store...")
        self._vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            self._embeddings,
            allow_dangerous_deserialization=True
        )

        print("  ├─ Loading LLM (this may take a minute)...")
        self._load_llm()

        print("  ├─ Building retriever (with reranker)...")
        self._retriever = get_retriever(self._vectorstore)

        # Build QA chain
        self._build_chain()

        self._initialized = True
        print("✓ QA service initialized\n")

        from .config import print_config
        print_config()


    def update_llm_params(self, overrides: dict):
        """
        Update LLM parameters at runtime and rebuild the QA chain.
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized; call initialize() first.")

        print("♻️ Rebuilding LLM with new parameters...")
        self._load_llm(overrides)
        self._build_chain()
        print("✓ LLM parameters updated\n")

    def current_llm_params(self) -> Dict:
        """Return current LLM parameters (base config + overrides)."""
        merged = {
            "model_path": self._llm_overrides.get("model_path", GGUF_MODEL_PATH),
            "n_ctx": self._llm_overrides.get("n_ctx", N_CTX),
            "n_threads": self._llm_overrides.get("n_threads", N_THREADS),
            "n_gpu_layers": self._llm_overrides.get("n_gpu_layers", N_GPU_LAYERS),
            "n_batch": self._llm_overrides.get("n_batch", N_BATCH),
            "temperature": self._llm_overrides.get("temperature", TEMPERATURE),
            "max_tokens": self._llm_overrides.get("max_tokens", MAX_TOKENS),
            "top_p": self._llm_overrides.get("top_p", TOP_P),
            "repeat_penalty": self._llm_overrides.get("repeat_penalty", REPEAT_PENALTY),
        }
        return merged

    def ask(self, question: str, history: Optional[List[Dict]] = None) -> Dict:
        """
        Ask a question (optionally with chat history) and get an answer.

        Args:
            question: User query
            history: Optional list of prior messages:
                [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            {
                "answer": str,
                "sources": List[Dict]
            }
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized; call initialize() first.")

        try:
            history_block = self._format_history(history)
            query_text = question
            if history_block:
                query_text = f"{history_block}\nUser: {question}"

            result = self.qa_chain.invoke({"query": query_text})

            # Clean the answer
            answer = result["result"].split("<|eot_id|>")[0].strip()

            # Extract sources
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", None)
                })

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            raise Exception(f"Error while answering question: {str(e)}")

    def health_check(self) -> Dict:
        """Health check."""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "model_loaded": self.qa_chain is not None,
            "config": Settings.to_dict(),  # Return current configuration
            "model_info": Settings.get_model_info()  # Include model file info
        }

# Global singleton
qa_service = QAService()
