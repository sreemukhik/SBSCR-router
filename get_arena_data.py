import requests
import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN').strip()

# Based on research, this is the correct direct URL for the Chatbot Arena conversations parquet
url = "https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/resolve/main/data/train-00000-of-00001.parquet"

def get_data():
    print(f"🚀 Attempting direct download from LMSYS Arena...")
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 403:
            print("❌ ACCESS FORBIDDEN: This token does not have 'Write' access or hasn't synced yet.")
            return
        
        response.raise_for_status()
        
        temp_path = "data/arena_temp.parquet"
        os.makedirs("data", exist_ok=True)
        
        print("📥 Downloading parquet file...")
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("📊 Processing parquet data...")
        # We use pandas to read the parquet since the 'datasets' library is having DLL issues
        df = pd.read_parquet(temp_path)
        
        # In this dataset, the column 'conversation_a' or 'conversation_b' contains a list of dicts
        # format: [{'content': '...', 'role': 'user'}, ...]
        extracted_data = []
        for _, row in df.iterrows():
            conv = row.get('conversation_a', [])
            if conv and len(conv) > 0:
                user_prompt = conv[0]['content']
                extracted_data.append({
                    "query": user_prompt,
                    "domain": "general"
                })
            
            if len(extracted_data) >= 2000:
                break
        
        output_path = "data/prompts_to_calibrate.json"
        with open(output_path, 'w') as f:
            json.dump(extracted_data, f, indent=2)
            
        print(f"✅ SUCCESS! Saved {len(extracted_data)} real queries from LMSYS to {output_path}")
        
        # Cleanup
        os.remove(temp_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    get_data()
