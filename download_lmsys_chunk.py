from huggingface_hub import hf_hub_download
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    print("Downloading LMSYS Parquet...")
    path = hf_hub_download(
        repo_id="lmsys/lmsys-chat-1m",
        filename="data/train-00000-of-00006-fe1acc5d10a9f0e2.parquet",
        repo_type="dataset",
        token=token
    )
    print(f"Downloaded to {path}")
except Exception as e:
    print(f"Error: {e}")
