from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

api = HfApi(token=token)
try:
    info = api.dataset_info("lmsys/lmsys-chat-1m")
    print(f"Info: {info}")
    files = api.list_repo_files("lmsys/lmsys-chat-1m", repo_type="dataset")
    print(f"Files: {files}")
except Exception as e:
    print(f"Error: {e}")
