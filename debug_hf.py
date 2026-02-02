import os
import sys
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    from datasets import load_dataset
    print("Import successful")
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train", streaming=True, token=token)
    print("Stream opened")
    it = iter(ds)
    first = next(it)
    print("Found data!")
    print(first.keys())
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
