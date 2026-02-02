import json
import os

input_path = "data/alpaca_data.json"
output_path = "data/prompts_to_calibrate.json"

try:
    with open(input_path, 'r') as f:
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
    with open(output_path, 'w') as f:
        json.dump(extracted, f, indent=2)
    print(f"Successfully saved {len(extracted)} prompts to {output_path}")

except Exception as e:
    print(f"Error: {e}")
