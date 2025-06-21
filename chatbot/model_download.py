from huggingface_hub import hf_hub_download

# Downloads the Q4_0 quantized GGUF (approx 1.98 GB)
file_path = hf_hub_download(
    repo_id="Aryanne/Orca-Mini-3B-gguf",
    filename="q4_0-orca-mini-3b.gguf",
    repo_type="model",
    cache_dir="models",     # local cache dir
)
print("Downloaded model to:", file_path)
