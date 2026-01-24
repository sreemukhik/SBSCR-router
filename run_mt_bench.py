# run_mt_bench.py - Custom MT-Bench Runner (No OpenAI SDK)
"""
Runs MT-Bench questions against your local server.
Uses raw HTTP requests - NO OpenAI SDK required.
"""

import json
import requests
import time
from pathlib import Path

# Configuration
SERVER_URL = "http://localhost:8000/v1/chat/completions"
QUESTIONS_FILE = "data/mt_bench/question.jsonl"
OUTPUT_FILE = "data/mt_bench/model_answer/sbscr-auto.jsonl"
MAX_TOKENS = 1024

def load_questions():
    """Load MT-Bench questions."""
    questions = []
    with open(QUESTIONS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def call_server(messages, max_tokens=MAX_TOKENS):
    """Call our local server (no OpenAI SDK needed)."""
    payload = {
        "model": "sbscr-auto",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    response = requests.post(SERVER_URL, json=payload, timeout=300)
    
    if response.status_code != 200:
        return f"$ERROR$ {response.status_code}: {response.text}"
    
    return response.json()["choices"][0]["message"]["content"]

def generate_answers(questions):
    """Generate answers for all questions."""
    results = []
    
    for i, q in enumerate(questions):
        print(f"\n📝 Question {i+1}/{len(questions)} (ID: {q['question_id']}, Category: {q.get('category', 'unknown')})")
        
        turns = q.get("turns", [])
        answers = []
        conversation = []
        
        for turn_idx, turn_prompt in enumerate(turns):
            print(f"   Turn {turn_idx + 1}: {turn_prompt[:50]}...")
            
            # Build conversation
            conversation.append({"role": "user", "content": turn_prompt})
            
            # Call server
            start = time.time()
            response = call_server(conversation)
            latency = time.time() - start
            
            print(f"   ✅ Response: {response[:100]}... ({latency:.1f}s)")
            
            # Add to conversation for multi-turn
            conversation.append({"role": "assistant", "content": response})
            answers.append(response)
        
        # Save result
        result = {
            "question_id": q["question_id"],
            "answer_id": f"sbscr-{q['question_id']}-{int(time.time())}",
            "model_id": "sbscr-auto",
            "choices": [{"index": 0, "turns": answers}],
            "tstamp": time.time()
        }
        results.append(result)
    
    return results

def save_results(results):
    """Save answers to JSONL file."""
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n💾 Saved {len(results)} answers to {OUTPUT_FILE}")

def main():
    print("="*60)
    print("🏆 MT-BENCH CUSTOM RUNNER (No OpenAI SDK)")
    print("="*60)
    print(f"Server: {SERVER_URL}")
    print(f"Questions: {QUESTIONS_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("="*60)
    
    # Load questions
    questions = load_questions()
    print(f"\n📊 Loaded {len(questions)} questions")
    
    # Generate answers
    results = generate_answers(questions)
    
    # Save
    save_results(results)
    
    print("\n✅ Done! You can now review the answers or run judgment.")

if __name__ == "__main__":
    main()
