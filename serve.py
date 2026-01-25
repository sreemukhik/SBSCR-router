# serve.py - SBSCR Enterprise Router v5
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import time
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="SBSCR Enterprise Router (Multi-Provider)")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SBSCR Router Playground</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #e2e8f0; max-width: 900px; margin: 0 auto; padding: 20px; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; }
            h1 { text-align: center; background: linear-gradient(to right, #60a5fa, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px; }
            #chat-container { flex: 1; overflow-y: auto; border: 1px solid #334155; border-radius: 12px; padding: 20px; background: #1e293b; display: flex; flex-direction: column; gap: 15px; }
            .message { padding: 12px 16px; border-radius: 12px; max-width: 80%; line-height: 1.5; }
            .user { background: #2563eb; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
            .assistant { background: #334155; color: #f1f5f9; align-self: flex-start; border-bottom-left-radius: 2px; border: 1px solid #475569; }
            .meta { font-size: 0.75rem; color: #94a3b8; margin-top: 8px; display: flex; align-items: center; gap: 10px; }
            .badge { background: #0f172a; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155; }
            .input-group { margin-top: 20px; display: flex; gap: 10px; }
            input { flex: 1; padding: 14px; border-radius: 8px; border: 1px solid #475569; background: #1e293b; color: white; font-size: 1rem; outline: none; transition: border-color 0.2s; }
            input:focus { border-color: #60a5fa; }
            button { padding: 0 24px; background: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 1rem; transition: background 0.2s; }
            button:hover { background: #1d4ed8; }
            button:disabled { background: #475569; cursor: not-allowed; }
            code { background: #0f172a; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em; }
            pre { background: #0f172a; padding: 10px; border-radius: 8px; overflow-x: auto; margin: 10px 0; }
        </style>
    </head>
    <body onload="document.getElementById('user-input').focus()">
        <h1>SBSCR Enterprise Router</h1>
        <div id="chat-container">
            <div class="message assistant">
                Hello! I am the <b>SBSCR Router</b> (v5).<br><br>
                I automatically route your prompts to the best open-source model (Llama 3, Qwen, DeepSeek, etc.) based on complexity and intent.<br><br>
                Try asking me a coding question or a logic puzzle!
            </div>
        </div>
        <div class="input-group">
            <input type="text" id="user-input" placeholder="Type a message..." autocomplete="off">
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>

        <script>
            const input = document.getElementById('user-input');
            const chatContainer = document.getElementById('chat-container');
            const sendBtn = document.getElementById('send-btn');

            input.addEventListener('keypress', function (e) {
                if (e.key === 'Enter' && !sendBtn.disabled) {
                    sendMessage();
                }
            });

            async function sendMessage() {
                const text = input.value.trim();
                if (!text) return;

                // Add User Message
                addMessage(text, 'user');
                input.value = '';
                setLoading(true);

                try {
                    const start = Date.now();
                    const response = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            messages: [{ role: "user", content: text }],
                            model: "sbscr-auto"
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) throw new Error(data.error.message || "Unknown error");
                    if (!data.choices) throw new Error("Invalid response format");

                    const reply = data.choices[0].message.content;
                    const meta = {
                        model: data.model,
                        latency: data.usage.total_latency_ms,
                        provider_model: data.usage.provider_model
                    };
                    
                    addMessage(reply, 'assistant', meta);
                } catch (e) {
                    addMessage("Error: " + e.message, 'assistant');
                } finally {
                    setLoading(false);
                    // Re-focus input for speed (removed for mobile usability)
                    // input.focus(); 
                }
            }

            function addMessage(text, role, meta = null) {
                const div = document.createElement('div');
                div.className = `message ${role}`;
                
                // Simple markdown-like formatting
                let formatted = text
                    .replace(/</g, "&lt;").replace(/>/g, "&gt;") // Escape HTML
                    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>') // Code blocks
                    .replace(/`([^`]+)`/g, '<code>$1</code>') // Inline code
                    .replace(/\\n/g, '<br>'); // Newlines

                div.innerHTML = formatted;

                if (meta) {
                    const metaDiv = document.createElement('div');
                    metaDiv.className = 'meta';
                    metaDiv.innerHTML = `
                        <span class="badge">🤖 ${meta.model}</span>
                        <span class="badge">⚡ ${meta.latency}ms</span>
                        <span style="opacity:0.6">via ${meta.provider_model}</span>
                    `;
                    div.appendChild(metaDiv);
                }

                chatContainer.appendChild(div);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function setLoading(isLoading) {
                sendBtn.disabled = isLoading;
                sendBtn.innerText = isLoading ? '...' : 'Send';
                input.disabled = isLoading;
                if (!isLoading) input.focus();
            }
        </script>
    </body>
    </html>
    """

# Initialize Components (Lazy Load for fast container startup)
import asyncio

router = None
providers = None
is_loading = True

async def initialize_heavy_components():
    global router, providers, is_loading
    print(f"INFO Startup: Starting background initialization...")
    
    # Lazy imports to save memory/startup time
    from sbscr.routers.sbscr import SBSCRRouter
    from sbscr.providers.base import ProviderRegistry
    from sbscr.providers.groq_provider import GroqProvider
    from sbscr.providers.huggingface_provider import HuggingFaceProvider
    from sbscr.providers.google_provider import GoogleProvider

    # Initialize Provider Registry
    try:
        temp_providers = ProviderRegistry()
        temp_providers.register(GroqProvider())
        temp_providers.register(HuggingFaceProvider())
        temp_providers.register(GoogleProvider())
        providers = temp_providers
        print("INFO Startup: Providers registered successfully.")
    except Exception as e:
        print(f"WARN Startup: Provider init failed: {e}")

    # Initialize Router (Heavy ML Load)
    try:
        print("INFO Startup: Loading ML Models (XGBoost/LSH)...")
        # Simulate small delay if needed or real heavy load
        router = SBSCRRouter()
        print("INFO Startup: Router initialized successfully.")
    except Exception as e:
        print(f"CRITICAL Startup: Router failed to load: {e}")
    
    is_loading = False
    print("INFO Startup: Initialization Complete.")

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("🏢 SBSCR ENTERPRISE ROUTER - 100% FREE TIER")
    print("="*60)
    print("\n📊 Model Output Mapping:")
    print("   - Claude Sonnet  -> Groq: Llama 3.1 70B")
    print("   - GPT-4 Turbo    -> HF: Qwen 2.5 72B")
    print("   - DeepSeek Coder -> HF: DeepSeek V2 Lite")
    print("\n🔑 Required API Keys (FREE):")
    print(f"   GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
    print(f"   HF_TOKEN:     {'✅ Set' if os.getenv('HF_TOKEN') else '❌ Missing'}")
    print(f"   GOOGLE_API_KEY:{'✅ Set' if os.getenv('GOOGLE_API_KEY') else '❌ Missing'}")
    print("="*60 + "\n")
    
    # Helper to check if we are in a cloud (Render) environment
    # or just local. In both cases, background loading is safer.
    asyncio.create_task(initialize_heavy_components())
    print("INFO Startup: Server is compliant and ready to bind port.")

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
async def chat_completions(request: ChatCompletionRequest):
    if is_loading:
        raise HTTPException(503, "Server is still initializing ML models. Please retry in 30 seconds.")
        
    if not request.messages:
        raise HTTPException(400, "No messages provided")
    
    query = request.messages[-1].content
    
    # Step 1: Route using Enterprise Router (Detailed)
    start = time.time()
    try:
        if router: 
            routing_result = router.route_detailed(query)
        else:
            # Emergency router unavailable fallback
            raise Exception("Router instance is None")
    except Exception as e:
        print(f"Router error: {e}, falling back to default")
        routing_result = {"model": "sbscr-sota", "metrics": {"error": str(e)}}
        
    selected_model = routing_result["model"]
    routing_metrics = routing_result["metrics"]
    routing_latency = (time.time() - start) * 1000
    
    # Step 2: Map to provider model
    provider_model = MODEL_MAP.get(selected_model, "llama-3.1-8b")
    
    # Step 3: Execute via provider
    try:
        inference_start = time.time()
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        
        if not providers:
             raise HTTPException(503, "Providers not initialized")
             
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
        "status": "loading" if is_loading else "ok",
        "providers_loaded": providers is not None,
        "router_loaded": router is not None
    }
