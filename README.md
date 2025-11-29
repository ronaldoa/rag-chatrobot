# 🦙 Llama 3 Local Knowledge-Base QA

High-performance local RAG (Retrieval Augmented Generation) built with LlamaCpp + GGUF + FAISS + a reranker. Runs fully on CPU with optional GPU acceleration and ships with both a Gradio web UI and REST API.

## ✨ Features
- 🚀 **Fast inference**: LlamaCpp + GGUF quantized models
- 🎯 **Accurate retrieval**: Two-stage flow (FAISS coarse + reranker fine)
- 💻 **CPU-first**: Works without a GPU (GPU layers optional)
- 🌐 **Two interfaces**: Gradio web UI + RESTful API
- 📚 **Multi-format docs**: TXT, PDF, DOCX, CSV, HTML
- 🔌 **Easy integration**: Standard REST endpoints for any client

## 🏗️ Architecture
User → Gradio UI / API → QA Service  
  ├─ FAISS retrieval (coarse top-20)  
  ├─ Reranker (fine top-3)  
  └─ LlamaCpp generation

## 📦 Installation

1) Clone the project
```bash
git clone <your-repo-url>
cd llama3-chatbot-hybrid
```

2) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Download a model  
Grab a GGUF file from [Hugging Face](https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF):
- Recommended: `llama-3.1-8b-instruct-q4_k_m.gguf` (~4.9GB)
- Smaller: `llama-3.1-8b-instruct-q3_k_m.gguf` (~3.3GB)
- Higher quality: `llama-3.1-8b-instruct-q5_k_m.gguf` (~5.8GB)

Place it in `models/`.

5) Configure environment
```bash
cp .env.example .env
# Edit .env as needed
```

## 🚀 Quick Start

1) Prepare documents  
Put files in `data/`:
```
data/
├── document1.pdf
├── document2.txt
└── notes.docx
```

2) Build the vector store
```bash
python ingest.py
```

3) Start the service
```bash
python app.py
```

4) Access
- Web UI: http://localhost:7860/
- API docs: http://localhost:7860/docs

## 📖 Usage

### Web UI
1. Open http://localhost:7860/  
2. Ask a question  
3. View the answer and source snippets  

### API examples
Python
```python
import requests

resp = requests.post(
    "http://localhost:7860/api/chat",
    json={"message": "What is the main content of the document?"}
)
print(resp.json()["answer"])
```

curl
```bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the main content of the document?"}'
```

JavaScript
```javascript
fetch('http://localhost:7860/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({message: 'What is the main content of the document?'})
})
.then(r => r.json())
.then(data => console.log(data.answer));
```

## ⚙️ Configuration
Edit `.env` to adjust parameters:

| Key | Description | Default |
| --- | --- | --- |
| `N_CTX` | Context window | 4096 |
| `N_THREADS` | CPU threads | 8 |
| `N_GPU_LAYERS` | GPU layers (0 = CPU only) | 0 |
| `INITIAL_K` | FAISS coarse results | 20 |
| `FINAL_K` | Reranker results | 3 |
| `TEMPERATURE` | Sampling temperature | 0.7 |
| `MAX_TOKENS` | Max generated tokens | 512 |

## 🔧 GPU Acceleration (optional)

CUDA (NVIDIA)
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
```
Set in `.env`:
```
N_GPU_LAYERS=35
```

Metal (Apple Silicon)
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
```

## 📊 Performance Reference

| Setup | Throughput | Memory |
| --- | --- | --- |
| CPU (8 cores) | 8-12 tok/s | 6-7GB |
| GPU (RTX 3060) | 25-30 tok/s | 6GB VRAM |
| M1 Pro (Metal) | 15-20 tok/s | 8GB |

## 🐛 FAQ
- **Model fails to load?** Check `GGUF_MODEL_PATH` in `.env`.  
- **Out of memory?** Use a smaller quantized model (Q3_K_M or Q2_K).  
- **Inaccurate answers?** Tune `INITIAL_K` / `FINAL_K` or adjust chunking parameters.  
- **Slow inference?** Enable GPU layers or reduce `MAX_TOKENS`.  

## 📂 Project Structure
```
llama3-chatbot-hybrid/
├── data/              # Raw documents
├── models/            # GGUF models
├── vector_store/      # Vector database
├── src/               # Core code
│   ├── config.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── retriever.py
│   ├── qa_service.py
│   └── utils.py
├── api/               # API endpoints
│   ├── chat.py
│   ├── documents.py
│   └── models.py
├── ui/                # Gradio UI
│   └── gradio_interface.py
├── app.py             # Main app
├── ingest.py          # Document ingestion
├── requirements.txt
├── .env.example
└── README.md
```

## 🤝 Contributing
Issues and PRs are welcome!

## 📄 License
MIT License

---

### .gitignore highlights
```
# Python
__pycache__/
*.py[cod]
*.egg-info/

# Virtual env
venv/

# Large files (ignored)
models/*.gguf
vector_store/
data/*.pdf
data/*.docx
data/*.csv
!data/sample.txt

# Env files
.env
.env.local

# Logs
logs/
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
```
