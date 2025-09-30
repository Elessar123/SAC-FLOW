from transformers import AutoTokenizer, AutoModel

model_name = "openai/clip-vit-large-patch14"
print(f" {model_name}...")

from huggingface_hub import snapshot_download

# snapshot_download(repo_id="openai/clip-vit-large-patch14", repo_type="model")

# /
AutoTokenizer.from_pretrained(model_name)
# 
# AutoModel.from_pretrained(model_name)

print("")