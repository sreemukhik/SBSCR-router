"""
Gemini Calibration Script for SBSCR Router.
Uses Gemini 1.5 Pro to "teach" the router what structural complexity looks like.
Outputs a calibrated dataset with golden labels for XGBoost training.
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configuration
INPUT_DATA_PATH = "data/prompts_to_calibrate.json"
OUTPUT_DATA_PATH = "data/calibrated_dataset.json"
GEMINI_MODEL = "gemini-flash-latest" # Updated to working model
MAX_SAMPLES = 2000 

# Set up Gemini
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel(GEMINI_MODEL)

CALIBRATION_PROMPT = """Analyze the structural complexity of the following user query. 
We are building a router that needs to decide if a query is 'Simple' (solvable by a 3B-8B model) or 'Complex' (requires 70B+ or GPT-4/Gemini Pro level).

Query: {query}

Provide your analysis in EXACTLY this JSON format:
{{
  "reasoning_steps": <estimated_steps_to_solve>,
  "logical_density": <1-5 scale of boolean/branching complexity>,
  "knowledge_breadth": <1-5 scale of domain knowledge required>,
  "difficulty_score": <1-10 scale where 1 is trivial and 10 is research-level>,
  "is_reasoning_heavy": <true/false>,
  "is_code_heavy": <true/false>,
  "recommended_model_tier": "<tiny|small|medium|large>"
}}
"""

def get_base_prompts():
    """Create or load a set of diverse prompts to calibrate."""
    if os.path.exists(INPUT_DATA_PATH):
        with open(INPUT_DATA_PATH, 'r') as f:
            return json.load(f)
    
    # Bootstrap common categories if no input exists
    print("🌱 No input dataset found. Bootstrapping diverse prompts...")
    prompts = [
        {"query": "What is 2+2?", "domain": "general"},
        {"query": "Write a hello world in Python.", "domain": "code"},
        {"query": "Explain quantum entanglement to a 5-year old.", "domain": "science"},
        {"query": "Write a high-performance concurrent hash map in C++ with lock-free reads.", "domain": "code"},
        {"query": "Solve the integral of x^2 * e^x from 0 to infinity.", "domain": "math"},
        {"query": "What are the geopolitical implications of the 1974 revolution in Ethiopia?", "domain": "history"},
        {"query": "Write a script to scrape a website and save it to a CSV.", "domain": "code"},
        {"query": "How do I boil an egg?", "domain": "general"},
        {"query": "Design a distributed system for a global social media platform with sub-second latency.", "domain": "system_design"},
        {"query": "If I have 3 apples and you have 2, how many apples do we have?", "domain": "math"}
    ]
    
    os.makedirs("data", exist_ok=True)
    with open(INPUT_DATA_PATH, 'w') as f:
        json.dump(prompts, f, indent=2)
    return prompts

def calibrate():
    prompts = get_base_prompts()
    # Limit to MAX_SAMPLES
    prompts = prompts[:MAX_SAMPLES]
    
    print(f"🚀 Starting Gemini Calibration for {len(prompts)} samples...")
    calibrated_data = []
    
    for item in tqdm(prompts):
        query = item['query']
        try:
            # Simple manual retry for 429
            for attempt in range(3):
                try:
                    response = model.generate_content(
                        CALIBRATION_PROMPT.format(query=query)
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        time.sleep(20) # Wait longer for rate limits
                    else:
                        raise e

            analysis = json.loads(response.text)
            
            # Combine original data with Gemini's analysis
            calibrated_item = {
                **item,
                "gemini_analysis": analysis,
                "golden_score": analysis.get("difficulty_score", 5) / 10.0,
                "timestamp": time.time()
            }
            calibrated_data.append(calibrated_item)
            
            # Save progress every 10 samples
            if len(calibrated_data) % 10 == 0:
                with open(OUTPUT_DATA_PATH, 'w') as f:
                    json.dump(calibrated_data, f, indent=2)
                print(f" (Progress saved: {len(calibrated_data)} items)")
            
            # Respect rate limits (simple sleep)
            time.sleep(2) # 2s sleep between successful calls
            
        except Exception as e:
            continue
            
    # Save the calibrated dataset
    with open(OUTPUT_DATA_PATH, 'w') as f:
        json.dump(calibrated_data, f, indent=2)
        
    print(f"✅ Calibration complete. Saved to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    calibrate()
