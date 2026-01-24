# FREE_TIER_SETUP.md - Zero-Cost Production Setup

## 🆓 Free Alternatives for All Models

### Option 1: Local Models (Ollama) - RECOMMENDED
Run models on your own machine for FREE.

#### Install Ollama
```bash
# Windows: Download from https://ollama.ai/download
# Or use WSL/Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Download Free Models
```bash
# SOTA Tier (replaces Claude/GPT-4)
ollama pull llama3.1:70b         # Best reasoning (13GB)
ollama pull qwen2.5:32b          # Great for coding (19GB)

# High Performance Tier
ollama pull llama3.1:8b          # Fast, good quality (4.7GB)
ollama pull mistral:7b           # Balanced (4.1GB)

# Code Specialists
ollama pull deepseek-coder-v2    # Best free code model (8.9GB)
ollama pull codellama:13b        # Alternative (7.4GB)

# Cheap/Fast Tier
ollama pull phi3:mini            # Tiny, fast (2.3GB)
ollama pull gemma2:2b            # Ultra-fast (1.6GB)
```

#### Update Your Router Config
Replace `data/models.yaml` with local endpoints:

```yaml
models:
  # SOTA Tier (Free Local)
  llama3.1-70b:
    provider: ollama
    endpoint: http://localhost:11434
    cluster: sota
    context_window: 128000
    price_in: 0.0  # FREE!
    price_out: 0.0
    reasoning: 85.0
    coding: 82.0
  
  qwen2.5-32b:
    provider: ollama
    endpoint: http://localhost:11434
    cluster: sota
    context_window: 32000
    price_in: 0.0
    price_out: 0.0
    reasoning: 83.0
    coding: 88.0

  # High Performance (Free Local)
  llama3.1-8b:
    provider: ollama
    endpoint: http://localhost:11434
    cluster: high_perf
    context_window: 128000
    price_in: 0.0
    price_out: 0.0
    reasoning: 75.0
    coding: 70.0

  # Code Specialist (Free Local)
  deepseek-coder-v2:
    provider: ollama
    endpoint: http://localhost:11434
    cluster: fast_code
    context_window: 16000
    price_in: 0.0
    price_out: 0.0
    reasoning: 70.0
    coding: 90.0

  # Cheap/Fast (Free Local)
  phi3-mini:
    provider: ollama
    endpoint: http://localhost:11434
    cluster: cheap_chat
    context_window: 128000
    price_in: 0.0
    price_out: 0.0
    reasoning: 60.0
    coding: 55.0
```

---

### Option 2: Free API Tiers (No Local Hardware Needed)

#### 1. **Groq** (FREE, Super Fast)
- **Free Tier**: 14,400 requests/day
- **Models**: Llama 3.1 70B, Mixtral, Gemma
- **Speed**: 500+ tokens/sec (fastest free option)
- **Sign up**: https://console.groq.com/

```bash
export GROQ_API_KEY="gsk_..."
```

#### 2. **Together AI** (FREE $25 Credit)
- **Free Tier**: $25 credit (lasts months)
- **Models**: Llama 3.1, Qwen, DeepSeek Coder
- **Sign up**: https://api.together.xyz/

```bash
export TOGETHER_API_KEY="..."
```

#### 3. **Hugging Face Inference API** (FREE)
- **Free Tier**: Rate-limited but unlimited
- **Models**: Thousands of open models
- **Sign up**: https://huggingface.co/

```bash
export HF_TOKEN="hf_..."
```

#### 4. **Google AI Studio** (FREE)
- **Free Tier**: 60 requests/minute
- **Models**: Gemini 1.5 Flash (free forever)
- **Sign up**: https://makersuite.google.com/

```bash
export GOOGLE_API_KEY="..."
```

---

## 🚀 Recommended Free Setup (Best Performance)

### Hybrid Approach: Groq + Ollama

**For Cloud (Fast, Free API)**:
- Use **Groq** for SOTA tier (Llama 3.1 70B)
- Use **Together AI** for code (DeepSeek Coder)

**For Local (Privacy, Unlimited)**:
- Use **Ollama** for everything else

### Updated Free Server (`serve_free.py`)

```python
# serve_free.py - 100% Free Production Server
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import time
import os
import requests

from sbscr.routers.sbscr import SBSCRRouter

app = FastAPI(title="SBSCR Free Router API")
router = SBSCRRouter()

# Free API Keys (Optional)
GROQ_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "sbscr-auto"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

def call_ollama(model_name: str, messages: List[Message], max_tokens: int):
    """Call local Ollama model (FREE)"""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"num_predict": max_tokens}
        }
    )
    return response.json()["message"]["content"]

def call_groq(model_name: str, messages: List[Message], max_tokens: int):
    """Call Groq API (FREE, 14k requests/day)"""
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set")
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens
        }
    )
    return response.json()["choices"][0]["message"]["content"]

def call_together(model_name: str, messages: List[Message], max_tokens: int):
    """Call Together AI (FREE $25 credit)"""
    if not TOGETHER_KEY:
        raise HTTPException(500, "TOGETHER_API_KEY not set")
    
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {TOGETHER_KEY}"},
        json={
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens
        }
    )
    return response.json()["choices"][0]["message"]["content"]

def call_model(model_name: str, messages: List[Message], max_tokens: int):
    """Route to appropriate FREE provider"""
    
    # Map your router's model names to free alternatives
    free_model_map = {
        # SOTA -> Groq (FREE, fast)
        "claude-3-5-sonnet": ("groq", "llama-3.1-70b-versatile"),
        "gpt-4-turbo": ("groq", "llama-3.1-70b-versatile"),
        "gemini-1.5-pro": ("groq", "llama-3.1-70b-versatile"),
        
        # High Perf -> Ollama (FREE, local)
        "llama-3-70b": ("ollama", "llama3.1:8b"),
        "mistral-large": ("ollama", "mistral:7b"),
        
        # Code -> Together AI (FREE credit)
        "deepseek-coder-v2": ("together", "deepseek-ai/deepseek-coder-33b-instruct"),
        
        # Cheap -> Ollama (FREE, local)
        "phi-3-mini": ("ollama", "phi3:mini"),
        "llama-3-8b": ("ollama", "llama3.1:8b"),
        "gemma-7b": ("ollama", "gemma2:2b"),
    }
    
    provider, free_model = free_model_map.get(model_name, ("ollama", "llama3.1:8b"))
    
    if provider == "ollama":
        return call_ollama(free_model, messages, max_tokens)
    elif provider == "groq":
        return call_groq(free_model, messages, max_tokens)
    elif provider == "together":
        return call_together(free_model, messages, max_tokens)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    query = request.messages[-1].content
    
    # Route
    selected_model = router.route(query)
    
    # Execute (FREE)
    try:
        response = call_model(selected_model, request.messages, request.max_tokens)
        return {
            "id": "chatcmpl-free",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(503, f"Error: {str(e)}")

if __name__ == "__main__":
    print("🆓 FREE MODE: Using Ollama + Groq + Together AI")
    print("   No API costs!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 💻 Hardware Requirements (Ollama)

- **Minimum**: 16GB RAM (run 7B models)
- **Recommended**: 32GB RAM (run 13B-32B models)
- **Ideal**: 64GB RAM + GPU (run 70B models)

If you have less RAM, use **Groq** (cloud, free) instead of Ollama.

---

## 🎯 Quick Start (Zero Cost)

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull llama3.1:8b
   ```

2. **Sign up for Groq** (FREE):
   ```bash
   # Visit: https://console.groq.com/
   export GROQ_API_KEY="gsk_..."
   ```

3. **Run Free Server**:
   ```bash
   python serve_free.py
   ```

4. **Test**:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Write a Python function"}]
     }'
   ```

**Total Cost**: $0.00 🎉
