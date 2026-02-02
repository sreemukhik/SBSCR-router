import requests
import json
import os

# A public, high-quality Python code dataset
url = "https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca/resolve/main/data/train-00000-of-00001.parquet"

try:
    print(f"Fetching public coding data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open("data/code_data.parquet", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Successfully downloaded coding dataset.")

except Exception as e:
    print(f"Error: {e}")
