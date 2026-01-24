# sbscr/providers/together_provider.py
"""
Together AI Provider - $25 free credit
4 models including best code models.
"""

import os
import requests
from typing import List, Dict
from sbscr.providers.base import BaseProvider, ProviderModel


class TogetherProvider(BaseProvider):
    """Together AI provider - $25 free credit on signup."""
    
    def __init__(self):
        super().__init__()
        self.name = "together"
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        
        # Define available models
        self.models = {
            # Code Specialist Tier (Best free code models)
            "deepseek-coder-33b": ProviderModel(
                name="deepseek-coder-33b",
                provider_id="deepseek-ai/deepseek-coder-33b-instruct",
                provider="together",
                cluster="fast_code",
                context_window=16000,
                speed_tokens_per_sec=100
            ),
            "codellama-34b": ProviderModel(
                name="codellama-34b",
                provider_id="codellama/CodeLlama-34b-Instruct-hf",
                provider="together",
                cluster="fast_code",
                context_window=16000,
                speed_tokens_per_sec=80
            ),
            
            # SOTA Tier
            "qwen2.5-72b": ProviderModel(
                name="qwen2.5-72b",
                provider_id="Qwen/Qwen2.5-72B-Instruct-Turbo",
                provider="together",
                cluster="sota",
                context_window=32000,
                speed_tokens_per_sec=150
            ),
            
            # High Performance Tier
            "mistral-7b": ProviderModel(
                name="mistral-7b",
                provider_id="mistralai/Mistral-7B-Instruct-v0.3",
                provider="together",
                cluster="high_perf",
                context_window=32000,
                speed_tokens_per_sec=200
            ),
        }
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def call(self, model_id: str, messages: List[Dict], max_tokens: int, temperature: float) -> str:
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not set. Get $25 free: https://api.together.xyz/")
        
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
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Together API error: {response.status_code} - {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]
