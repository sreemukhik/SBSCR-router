import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN').strip()
# Use the direct resolve URL
url = "https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/resolve/main/data/train-00000-of-00001.parquet"

try:
    print(f"Downloading from {url}...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 401:
        print("Unauthorized: Check your HF token.")
    elif response.status_code == 403:
        print("Forbidden: You likely need to accept terms at https://huggingface.co/datasets/lmsys/chatbot_arena_conversations")
    else:
        response.raise_for_status()
        with open("data/lmsys_arena.parquet", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")

except Exception as e:
    print(f"Error: {e}")
