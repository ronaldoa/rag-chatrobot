from llama_cpp import Llama
from pathlib import Path

# 换成你自己模型的真实路径
MODEL = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

print(">>> Loading model...")
llm = Llama(
    model_path=str(Path(MODEL)),
    n_ctx=512,
    n_gpu_layers=35,  # 关键：显式要求把一部分层丢到 GPU
    verbose=True,  # 关键：打开详细日志
)

print(">>> Model loaded, generating...")
out = llm("Hello", max_tokens=512)
print(">>> Done. Output:")
print(out)
