# sbscr/providers/huggingface_provider.py
"""
Hugging Face Provider - 100% FREE (Serverless Inference API)
Access to thousands of open models (Qwen, Mistral, DeepSeek, Phi)
Limit: Rate limited but usable for testing/prototyping.
"""

import os
import requests
from typing import List, Dict
from sbscr.providers.base import BaseProvider, ProviderModel


class HuggingFaceProvider(BaseProvider):
    """Hugging Face Inference API provider - FREE."""
    
    def __init__(self):
        super().__init__()
        self.name = "huggingface"
        self.api_key = os.getenv("HF_TOKEN")
        self.base_url_template = "https://api-inference.huggingface.co/models/{}"
        
        # Define available free serverless models
        # Note: Availability depends on model loading status on HF servers
        self.models = {
            # Code Models
            "deepseek-coder-v2": ProviderModel(
                name="deepseek-coder-v2",
                provider_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                provider="huggingface",
                cluster="fast_code",
                context_window=128000,
                speed_tokens_per_sec=40
            ),
             "starcoder2-15b": ProviderModel(
                name="starcoder2-15b",
                provider_id="bigcode/starcoder2-15b-instruct-v0.1", # Using instruct tuned variant if available, else fallback
                provider="huggingface",
                cluster="fast_code",
                context_window=16000,
                speed_tokens_per_sec=35
            ),
            
            # SOTA Replacement (Qwen 72B is often available on Inference API Pro, checking regular API)
            # Falling back to Qwen 2.5 32B for better free tier reliability
            "qwen2.5-72b": ProviderModel(
                name="qwen2.5-72b",
                provider_id="Qwen/Qwen2.5-72B-Instruct", 
                provider="huggingface",
                cluster="sota",
                context_window=32000,
                speed_tokens_per_sec=30
            ),
             "mistral-7b": ProviderModel(
                name="mistral-7b",
                provider_id="mistralai/Mistral-7B-Instruct-v0.3",
                provider="huggingface",
                cluster="high_perf",
                context_window=32000,
                speed_tokens_per_sec=60
            ),
            "phi-3-mini": ProviderModel(
                name="phi-3-mini",
                provider_id="microsoft/Phi-3-mini-4k-instruct",
                provider="huggingface",
                cluster="cheap_chat",
                context_window=4096,
                speed_tokens_per_sec=80
            )
        }
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def call(self, model_id: str, messages: List[Dict], max_tokens: int, temperature: float) -> str:
        if not self.api_key:
            raise ValueError("HF_TOKEN not set. Get free token: https://huggingface.co/settings/tokens")
        
        url = self.base_url_template.format(model_id)
        
        # Construct prompt based on model type or use raw messages if model supports chat template
        # HF Inference API for chat models usually accepts "inputs" as a string with templates
        # But some newer ones support simple payload. We will use the standard HF API format.
        
        # Simplified chat template application (very basic, ideally use tokenizer.apply_chat_template)
        # For simplicity in this lightweight provider, we construct a basic prompt string
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            # Handle "Model loading" 503 error
            if "estimated_time" in response.text:
                raise Exception(f"Model is loading (HF cold start). Try again in {response.json().get('estimated_time', 20)}s.")
            raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)
