from llama_cpp import Llama

llm = Llama(
    model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
)

out = llm(
    "Q: What is 2 + 2?\nA:",
    max_tokens=16,
    temperature=0.1,
    stop=["Q:", "\n\n"],
)
print(out["choices"][0]["text"])
