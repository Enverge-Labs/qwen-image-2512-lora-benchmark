"""Downloads the model and the training images into Hugging Face's cache and work/dataset."""
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen-Image-2512")
# Public 5-image dataset; content doesn't affect speed, resolution and batch size do.
snapshot_download("diffusers/dog-example", repo_type="dataset", local_dir="work/dataset")
