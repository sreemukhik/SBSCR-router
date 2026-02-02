from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    print(f"Loading alpaca-gpt4...")
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train", streaming=True, token=token)
    print("Success!")
    for i, row in enumerate(dataset):
        print(row.keys())
        if i >= 0: break
except Exception as e:
    print(f"Error: {e}")
