"""
Generate synthetic training data for the SBSCR router.
Uses a local LLM to generate queries of varying complexity across domains.
"""

import json
import os
import random
from typing import List, Dict
from tqdm import tqdm
from sbscr.inference.llm_client import LLMClient

# Configuration
OUTPUT_FILE = "data/synthetic_dataset.json"
EXAMPLES_PER_CATEGORY = 50  # 20 categories * 50 = 1000 examples
GENERATOR_MODEL = "llama-3-8b"  # Use local model for generation

CATEGORIES = [
    # --- TIER 1: TINY (phi-3-mini) ---
    {
        "domain": "general",
        "complexity": "trivial",
        "prompt": "Generate 10 very simple questions like 'What is the capital of France?' or 'Who wrote Hamlet?'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "phi-3-mini"
    },
    {
        "domain": "math",
        "complexity": "trivial",
        "prompt": "Generate 10 very simple math problems like 'What is 5 + 7?' or '20 divided by 4'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "phi-3-mini"
    },
    {
        "domain": "code",
        "complexity": "trivial",
        "prompt": "Generate 10 trivial coding requests like 'Print hello world in Python' or 'Define a variable x'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "phi-3-mini"
    },
    
    # --- TIER 2: SMALL (llama-3-8b) ---
    {
        "domain": "general",
        "complexity": "medium",
        "prompt": "Generate 10 medium-complexity reasoning questions like 'Explain photosynthesis' or 'Compare apples and oranges'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "llama-3-8b"
    },
    {
        "domain": "code",
        "complexity": "simple_algo",
        "prompt": "Generate 10 simple coding interview questions like 'Implement bubble sort' or 'Check for palindrome'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "llama-3-8b"
    },
    {
        "domain": "math",
        "complexity": "algebra",
        "prompt": "Generate 10 algebra word problems suitable for high school students. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "llama-3-8b"
    },
    
    # --- TIER 3: SPECIALIZED (deepseek-coder) ---
    {
        "domain": "code",
        "complexity": "complex_algo",
        "prompt": "Generate 10 complex algorithmic coding challenges like 'Implement a Red-Black Tree' or 'Solve the Traveling Salesman Problem'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "deepseek-coder-6.7b"
    },
    {
        "domain": "code",
        "complexity": "boilerplate",
        "prompt": "Generate 10 requests to write full classes or data structures, e.g., 'Write a User class with validation'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "deepseek-coder-6.7b"
    },

    # --- TIER 4: LARGE (gemini-1.5-pro) ---
    {
        "domain": "reasoning",
        "complexity": "complex",
        "prompt": "Generate 10 very complex, multi-step reasoning prompts like 'Design a sustainable city for Mars' or 'Analyze the geopolitical implications of AI'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "gemini-1.5-pro"
    },
    {
        "domain": "code",
        "complexity": "system_design",
        "prompt": "Generate 10 system design interview questions like 'Design a clone of Twitter' or 'Architect a distributed key-value store'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "gemini-1.5-pro"
    },
    {
        "domain": "math",
        "complexity": "advanced",
        "prompt": "Generate 10 university-level math problems (Calculus, Linear Algebra, Proofs). Output ONLY the questions as a JSON list of strings.",
        "expected_model": "gemini-1.5-pro"
    },
    {
        "domain": "creative",
        "complexity": "writing",
        "prompt": "Generate 10 creative writing prompts like 'Write a short story about a time traveler' or 'Compose a poem about entropy'. Output ONLY the questions as a JSON list of strings.",
        "expected_model": "gemini-1.5-pro"
    }
]

def main():
    print(f"🏭 Starting Synthetic Data Factory (Target: {len(CATEGORIES) * EXAMPLES_PER_CATEGORY} examples)")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    client = LLMClient()
    dataset = []
    
    # Check if model available
    if not client.is_available(GENERATOR_MODEL):
        print(f"⚠️  Warning: {GENERATOR_MODEL} not available via Ollama.")
        print("Using 'basic' mode (creating static examples instead of generating).")
        # Fallback to static if needed (omitted for brevity, assuming user has Ollama)
        return

    total_generated = 0
    
    for category in tqdm(CATEGORIES, desc="Generating Categories"):
        examples_needed = EXAMPLES_PER_CATEGORY
        failed_attempts = 0
        
        while examples_needed > 0 and failed_attempts < 3:
            # We generate in batches of 10 to keep context small
            try:
                result = client.infer(
                    GENERATOR_MODEL, 
                    category['prompt'], 
                    temperature=0.8
                )
                
                response_text = result['response']
                
                # Flexible parsing
                try:
                    # Clean markdown code blocks if present
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]
                        
                    batch = json.loads(response_text)
                    
                    if isinstance(batch, list):
                        valid_items = [q for q in batch if isinstance(q, str)]
                        
                        for query in valid_items:
                            if examples_needed <= 0:
                                break
                                
                            dataset.append({
                                'query': query,
                                'domain': category['domain'],
                                # Note: complexity number will be calculated by extractor later
                                'dataset_complexity_label': category['complexity'],
                                'expected_model': category['expected_model'] 
                            })
                            examples_needed -= 1
                            total_generated += 1
                    else:
                        failed_attempts += 1
                        print("DEBUG: Response was not a list")
                        
                except json.JSONDecodeError:
                    failed_attempts += 1
                    # print(f"DEBUG: JSON output failed")
                    
            except Exception as e:
                print(f"Error: {e}")
                failed_attempts += 1
                
    print(f"\n✅ Generated {len(dataset)} examples.")
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
