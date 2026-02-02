from huggingface_hub import hf_hub_download
import os
from dotenv import load_dotenv
import json

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    print("Downloading file via HF Hub...")
    # Chatbot Arena is large, let's try a smaller one or look for a specific file
    # For lmsys/chatbot_arena_conversations, let's see if we can get a sample
    # Actually, let's try alpaca-gpt4 which is simpler
    file_path = hf_hub_download(repo_id="vicgalle/alpaca-gpt4", filename="alpaca_gpt4_data.json", token=token)
    print(f"Downloaded to {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")
    
    # Process 2000
    extracted = []
    for item in data[:2000]:
        prompt = item.get('instruction', '')
        if item.get('input'):
            prompt += "\n" + item['input']
        extracted.append({'query': prompt, 'domain': 'general'})
        
    os.makedirs("data", exist_ok=True)
    with open("data/prompts_to_calibrate.json", 'w') as f:
        json.dump(extracted, f, indent=2)
    print("Successfully saved 2000 prompts to data/prompts_to_calibrate.json")
    
except Exception as e:
    print(f"Error: {e}")
