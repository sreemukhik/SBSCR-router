# sbscr/providers/google_provider.py
"""
Google AI Studio Provider - FREE
Gemini models with generous free tier.
"""

import os
from typing import List, Dict
import warnings
from sbscr.providers.base import BaseProvider, ProviderModel


class GoogleProvider(BaseProvider):
    """Google AI Studio provider - FREE tier with 15 requests/minute."""
    
    def __init__(self):
        super().__init__()
        self.name = "google"
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.genai = None
        
        if self.api_key:
            try:
                # Suppress FutureWarning from google.generativeai
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore") 
                    import google.generativeai as genai
                self.genai = genai
                self.genai.configure(api_key=self.api_key)
            except ImportError:
                self.genai = None
        
        # Define available models
        self.models = {
            # SOTA Tier
            "gemini-1.5-pro": ProviderModel(
                name="gemini-1.5-pro",
                provider_id="gemini-1.5-pro",
                provider="google",
                cluster="sota",
                context_window=2000000,
                speed_tokens_per_sec=50
            ),
            
            # High Performance / Fast Tier
            "gemini-1.5-flash": ProviderModel(
                name="gemini-1.5-flash",
                provider_id="gemini-1.5-flash",
                provider="google",
                cluster="high_perf",
                context_window=1000000,
                speed_tokens_per_sec=150
            ),
        }
    
    def is_available(self) -> bool:
        return self.api_key is not None and self.genai is not None
    
    def call(self, model_id: str, messages: List[Dict], max_tokens: int, temperature: float) -> str:
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set. Get free: https://makersuite.google.com/")
        if not self.genai:
            raise ValueError("google-generativeai not installed. Run: pip install google-generativeai")
        
        # Convert messages to Gemini format
        # Gemini uses a different format than OpenAI
        model = self.genai.GenerativeModel(model_id)
        
        # Build conversation history
        chat_history = []
        for msg in messages[:-1]:  # All but last
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})
        
        # Start chat with history
        chat = model.start_chat(history=chat_history)
        
        # Send final message
        response = chat.send_message(
            messages[-1]["content"],
            generation_config=self.genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
        
        return response.text
