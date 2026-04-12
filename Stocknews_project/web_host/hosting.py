
from huggingface_hub import HfApi
import os

api=HfApi(token=os.getenv("hf-api-key"))
repo_id = "Lokeshnathy/investment-stocks"
repo_type="space"

api.upload_folder(
    folder_path="PGP-AIML/Projects/Stocknews_project/deploy",
    repo_id=repo_id,
    repo_type=repo_type)
