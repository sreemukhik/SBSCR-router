# sbscr/providers/base.py
"""
Base provider interface and registry.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
import os


@dataclass
class ProviderModel:
    """A model available through a provider."""
    name: str              # Internal name (e.g., "llama-3.1-70b")
    provider_id: str       # Provider's model ID
    provider: str          # Provider name (groq, together, google)
    cluster: str           # SOTA, HIGH_PERF, FAST_CODE, CHEAP_CHAT
    context_window: int
    speed_tokens_per_sec: int
    is_free: bool = True


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self):
        self.name = "base"
        self.api_key = None
        self.models: Dict[str, ProviderModel] = {}
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass
    
    @abstractmethod
    def call(self, model_id: str, messages: List[Dict], max_tokens: int, temperature: float) -> str:
        """Make an API call to the provider."""
        pass
    
    def get_models(self) -> List[ProviderModel]:
        """Return list of available models."""
        return list(self.models.values())


class ProviderRegistry:
    """Registry of all available providers."""
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.model_to_provider: Dict[str, str] = {}
    
    def register(self, provider: BaseProvider):
        """Register a provider."""
        self.providers[provider.name] = provider
        for model in provider.get_models():
            self.model_to_provider[model.name] = provider.name
    
    def get_provider(self, model_name: str) -> Optional[BaseProvider]:
        """Get the provider for a given model."""
        provider_name = self.model_to_provider.get(model_name)
        if provider_name:
            return self.providers.get(provider_name)
        return None
    
    def call(self, model_name: str, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Route call to appropriate provider."""
        provider = self.get_provider(model_name)
        if not provider:
            raise ValueError(f"No provider found for model: {model_name}")
        if not provider.is_available():
            raise ValueError(f"Provider {provider.name} is not configured")
        
        # Get the provider's model ID
        model = provider.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found in provider {provider.name}")
        
        return provider.call(model.provider_id, messages, max_tokens, temperature)
    
    def list_available_models(self) -> List[str]:
        """List all available models across all configured providers."""
        models = []
        for provider in self.providers.values():
            if provider.is_available():
                models.extend([m.name for m in provider.get_models()])
        return models
