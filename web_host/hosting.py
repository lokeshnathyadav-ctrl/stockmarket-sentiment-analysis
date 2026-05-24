import os
from huggingface_hub import HfApi
api=HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "Lokeshnathy/INVESTMENT-STOCKS"
repo_type="space"
api.upload_folder(
    folder_path="deploy",
    repo_id=repo_id,
    repo_type=repo_type)
