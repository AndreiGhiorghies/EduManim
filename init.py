import os

from huggingface_hub import snapshot_download

# Run just one time
os.makedirs("./LLM/Models", exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    local_dir="./LLM/Models",
    allow_patterns=["qwen2.5-3b-instruct-q4_k_m.gguf"]
)

