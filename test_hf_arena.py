from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    print(f"Loading dataset with token: {token[:5]}...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train", streaming=True, token=token)
    print("Success!")
    for i, row in enumerate(dataset):
        print(row.keys())
        if i >= 0: break
except Exception as e:
    print(f"Error: {e}")
