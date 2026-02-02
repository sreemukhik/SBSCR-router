import requests
import json
import os

url = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data.json"

try:
    print(f"Fetching data from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print(f"Successfully fetched {len(data)} items")
    
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
