# serve.py - SBSCR Enterprise Router v5
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import time
import os
from dotenv import load_dotenv
load_dotenv()

from sbscr.routers.sbscr import SBSCRRouter
from sbscr.providers.base import ProviderRegistry
from sbscr.providers.groq_provider import GroqProvider
from sbscr.providers.huggingface_provider import HuggingFaceProvider
from sbscr.providers.google_provider import GoogleProvider

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SBSCR Enterprise Router (Multi-Provider)")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Components
router = SBSCRRouter()
providers = ProviderRegistry()

try:
    providers.register(GroqProvider())
    providers.register(HuggingFaceProvider())
    providers.register(GoogleProvider())
except Exception as e:
    print(f"Warning: Provider init failed: {e}")

# Optimized model mapping for Sub-Second response
MODEL_MAP = {
    "claude-3-5-sonnet": "llama-3.3-70b", # Groq (Fastest)
    "sbscr-sota": "llama-3.3-70b",        # Groq (Fastest)
    "gemini-1.5-pro": "gemini-1.5-pro",   # Google (Reliable)
    "llama-3-70b": "llama-3.3-70b",       # Groq
    "mistral-large": "mixtral-8x7b",      # Groq
    "mixtral-8x22b": "mixtral-8x7b",      # Groq
    "deepseek-coder-v2": "deepseek-coder-v2", # HF
    "starcoder2-15b": "starcoder2-15b",    # HF
    "llama-3-8b": "llama-3.1-8b",         # Groq (Instant)
    "phi-3-mini": "llama-3.1-8b",         # Groq (Redirect for speed)
    "gemma-7b": "gemini-1.5-flash",       # Google
    "haiku-3": "llama-3.1-8b",            # Groq
}

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "sbscr-auto"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

from fastapi import UploadFile, File
import io
from pypdf import PdfReader

@app.post("/v1/files/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"
            
        # Limit text for demo purposes
        word_count = len(extracted_text.split())
        
        return {
            "status": "success",
            "message": "Document parsed successfully",
            "filename": file.filename,
            "text": extracted_text[:10000], # Keep it manageable
            "context_added": word_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Parsing Error: {str(e)}")

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(400, "No messages provided")
    
    query = request.messages[-1].content
    
    # Step 1: Route using Enterprise Router (Detailed)
    start = time.time()
    routing_result = router.route_detailed(query)
    selected_model = routing_result["model"]
    routing_metrics = routing_result["metrics"]
    routing_latency = (time.time() - start) * 1000
    
    # Step 2: Map to provider model
    provider_model = MODEL_MAP.get(selected_model, "llama-3.1-8b")
    
    # Step 3: Execute via provider
    try:
        inference_start = time.time()
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        
        response = providers.call(
            provider_model,
            messages_dict,
            request.max_tokens,
            request.temperature
        )
        inference_latency = (time.time() - inference_start) * 1000
        
        return {
            "id": f"chatcmpl-enterprise-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(query) // 4,
                "completion_tokens": len(response) // 4,
                "total_tokens": (len(query) + len(response)) // 4,
                "routing_decision": selected_model,
                "provider_model": provider_model,
                "routing_latency_ms": round(routing_latency, 2),
                "inference_latency_ms": round(inference_latency, 2),
                "total_latency_ms": round(routing_latency + inference_latency, 2),
                "routing_analysis": routing_metrics
            }
        }
    except Exception as e:
        print(f"ERROR Provider {provider_model} failed: {e}")
        # Fallback chain
        intent = routing_result["metrics"].get("detected_intent")
        intent_conf = routing_result["metrics"].get("intent_confidence")
        fallback_chain = router.route_with_fallbacks(query, intent=intent, intent_conf=intent_conf)
        
        for fallback_model in fallback_chain[1:]:
            try:
                print(f"INFO Attempting fallback to {fallback_model}...")
                provider_fallback = MODEL_MAP.get(fallback_model, "llama-3.1-8b")
                messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
                res = providers.call(provider_fallback, messages_dict, request.max_tokens, request.temperature)
                
                # Success on fallback
                return {
                    "id": f"chatcmpl-fallback-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": fallback_model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": res
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "routing_latency_ms": round(routing_latency, 2),
                        "inference_latency_ms": 0, # Not tracked for fallback
                        "total_latency_ms": 0,
                        "routing_analysis": routing_metrics,
                        "fallback_used": True
                    }
                }
            except Exception as fe:
                print(f"ERROR Fallback {fallback_model} also failed: {fe}")
                continue
                
        raise HTTPException(500, f"All models failed: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "available_models": providers.list_available_models(),
        "providers": {
            "groq": os.getenv("GROQ_API_KEY") is not None,
            "huggingface": os.getenv("HF_TOKEN") is not None,
            "google": os.getenv("GOOGLE_API_KEY") is not None
        }
    }

if __name__ == "__main__":
    print("-" * 30)
    print("SBSCR ENTERPRISE ROUTER READY")
    print(f"Port 8000 | Keys: {'OK' if all([os.getenv('GROQ_API_KEY'), os.getenv('HF_TOKEN'), os.getenv('GOOGLE_API_KEY')]) else 'INCOMPLETE'}")
    print("-" * 30)
    uvicorn.run(app, host="0.0.0.0", port=8005)
