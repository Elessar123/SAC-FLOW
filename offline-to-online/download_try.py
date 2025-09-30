from transformers import AutoTokenizer, AutoModel

model_name = "openai/clip-vit-large-patch14"
print(f"正在下载模型 {model_name}...")

from huggingface_hub import snapshot_download

# snapshot_download(repo_id="openai/clip-vit-large-patch14", repo_type="model")

# 这行代码会自动下载模型/分词器文件并缓存到本地
AutoTokenizer.from_pretrained(model_name)
# 如果需要模型权重，也一并下载
# AutoModel.from_pretrained(model_name)

print("下载完成！")