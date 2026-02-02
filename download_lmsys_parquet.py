from huggingface_hub import HfApi, hf_hub_download
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    api = HfApi(token=token)
    repo_id = "lmsys/lmsys-chat-1m"
    files = api.list_repo_files(repo_id=repo_id)
    print(f"Files in {repo_id}: {files}")
    
    # Track down a parquet file
    parquet_files = [f for f in files if f.endswith('.parquet')]
    if not parquet_files:
        print("No parquet files found.")
    else:
        target_file = parquet_files[0]
        print(f"Downloading {target_file}...")
        path = hf_hub_download(repo_id=repo_id, filename=target_file, repo_type="dataset", token=token)
        print(f"Downloaded to: {path}")
        
except Exception as e:
    print(f"Error: {e}")
