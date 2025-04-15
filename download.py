from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="google/gemma-3-4b-it-qat-q4_0-gguf",
    filename="gemma-3-4b-it-q4_0.gguf",
    token="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"
)
print(f"Model downloaded to: {model_path}")
