# sbscr/providers/groq_provider.py
"""
Groq Provider - FREE, Super Fast (500+ tokens/sec)
5 models available in free tier.
"""

import os
import requests
from typing import List, Dict
from sbscr.providers.base import BaseProvider, ProviderModel


class GroqProvider(BaseProvider):
    """Groq API provider - FREE tier with 14,400 requests/day."""
    
    def __init__(self):
        super().__init__()
        self.name = "groq"
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Define available models (all FREE)
        self.models = {
            # SOTA Tier
            "llama-3.1-70b": ProviderModel(
                name="llama-3.1-70b",
                provider_id="llama-3.3-70b-versatile", # Updated from 3.1
                provider="groq",
                cluster="sota",
                context_window=128000,
                speed_tokens_per_sec=300
            ),
            "llama-3.3-70b": ProviderModel(
                name="llama-3.3-70b",
                provider_id="llama-3.3-70b-versatile",
                provider="groq",
                cluster="sota",
                context_window=128000,
                speed_tokens_per_sec=200
            ),
            
            # High Performance Tier
            "mixtral-8x7b": ProviderModel(
                name="mixtral-8x7b",
                provider_id="mixtral-8x7b-32768",
                provider="groq",
                cluster="high_perf",
                context_window=32000,
                speed_tokens_per_sec=500
            ),
            
            # Fast/Cheap Tier
            "llama-3.1-8b": ProviderModel(
                name="llama-3.1-8b",
                provider_id="llama-3.1-8b-instant", # Still valid
                provider="groq",
                cluster="cheap_chat",
                context_window=128000,
                speed_tokens_per_sec=750
            ),
            "gemma2-9b": ProviderModel(
                name="gemma2-9b",
                provider_id="gemma2-9b-it",
                provider="groq",
                cluster="cheap_chat",
                context_window=8192,
                speed_tokens_per_sec=600
            ),
        }
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def call(self, model_id: str, messages: List[Dict], max_tokens: int, temperature: float) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Get free key: https://console.groq.com/")
        
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]
