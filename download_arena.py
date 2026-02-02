from huggingface_hub import hf_hub_download
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    print("Downloading Chatbot Arena Parquet...")
    # List files first
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = api.list_repo_files("lmsys/chatbot_arena_conversations", repo_type="dataset")
    print(f"Files: {files}")
    parquet_files = [f for f in files if f.endswith('.parquet')]
    if parquet_files:
        path = hf_hub_download(
            repo_id="lmsys/chatbot_arena_conversations",
            filename=parquet_files[0],
            repo_type="dataset",
            token=token
        )
        print(f"Downloaded to {path}")
    else:
        print("No parquet files.")
except Exception as e:
    print(f"Error: {e}")
