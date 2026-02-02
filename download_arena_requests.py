import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')
url = "https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/resolve/main/data/train-00000-of-00001.parquet"

try:
    print(f"Downloading from {url}...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open("data/lmsys_arena.parquet", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete: data/lmsys_arena.parquet")

except Exception as e:
    print(f"Error: {e}")
