"""
LLM Client for running inference on different models.
Supports Ollama (local) and API-based models (Gemini).
"""

import os
import json
import subprocess
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Base class for LLM clients."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.ollama_client = OllamaClient()
        self.gemini_client = GeminiClient()
        
        # Map model names to clients
        self.model_routing = {
            'phi-3-mini': 'ollama',
            'llama-3-8b': 'ollama',
            'llama3:8b': 'ollama',
            'deepseek-coder-6.7b': 'ollama',
            'deepseek-coder': 'ollama',
            'gemini-1.5-pro': 'gemini',
            'gemini-pro': 'gemini',
        }
    
    def infer(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Run inference on specified model.
        
        Args:
            model: Model identifier
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dictionary with 'response', 'latency_ms', 'model'
        """
        client_type = self.model_routing.get(model.lower(), 'ollama')
        
        if client_type == 'ollama':
            return self.ollama_client.infer(model, prompt, **kwargs)
        elif client_type == 'gemini':
            return self.gemini_client.infer(model, prompt, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def is_available(self, model: str) -> bool:
        """Check if model is available."""
        client_type = self.model_routing.get(model.lower())
        
        if client_type == 'ollama':
            return self.ollama_client.is_model_available(model)
        elif client_type == 'gemini':
            return self.gemini_client.is_available()
        return False


class OllamaClient:
    """Client for Ollama local models."""
    
    def __init__(self):
        """Initialize Ollama client."""
        self.base_url = "http://localhost:11434"
        
        # Model name mappings
        self.model_map = {
            'phi-3-mini': 'phi3',
            'llama-3-8b': 'llama3:latest',  # Use latest since that's installed
            'llama3-8b': 'llama3:latest',
            'llama3': 'llama3:latest',
            'deepseek-coder-6.7b': 'deepseek-coder',
            'deepseek-coder': 'deepseek-coder:6.7b',
        }
    
    def infer(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Run inference via Ollama.
        
        Args:
            model: Model name
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        import time
        import requests
        
        # Map model name
        ollama_model = self.model_map.get(model.lower(), model)
        
        # Prepare request - using Ollama's generate API
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False
        }
        
        # Add options if provided
        if 'temperature' in kwargs or 'max_tokens' in kwargs:
            payload["options"] = {}
            if 'temperature' in kwargs:
                payload["options"]["temperature"] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                payload["options"]["num_predict"] = kwargs['max_tokens']
        
        # Run inference
        start = time.perf_counter()
        try:
            # Check if model exists, if not fallback to default
            if ':' not in ollama_model and ollama_model not in ['llama3', 'llama3:latest']:
                # Simplistic check - assume if it failed before or is unknown, try llama3
                pass 

            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # Model likely not found, try fallback to llama3:latest
                    print(f"⚠️  Model '{ollama_model}' not found. Falling back to 'llama3:latest'")
                    payload["model"] = "llama3:latest"
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=120
                    )
                    response.raise_for_status()
                    ollama_model = "llama3:latest (fallback)"
                else:
                    raise e
            
            end = time.perf_counter()
            
            result = response.json()
            
            return {
                'response': result.get('response', ''),
                'latency_ms': (end - start) * 1000,
                'model': ollama_model,
                'success': True
            }
        except Exception as e:
            end = time.perf_counter()
            return {
                'response': '',
                'latency_ms': (end - start) * 1000,
                'model': ollama_model,
                'success': False,
                'error': str(e)
            }
    
    def is_model_available(self, model: str) -> bool:
        """Check if Ollama model is available."""
        import requests
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                ollama_model = self.model_map.get(model.lower(), model)
                
                # Check if model exists
                for m in models:
                    if ollama_model in m.get('name', ''):
                        return True
            return False
        except:
            return False
    
    def list_models(self) -> list:
        """List available Ollama models."""
        import requests
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except:
            return []


class GeminiClient:
    """Client for Google Gemini API."""
    
    def __init__(self):
        """Initialize Gemini client."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        self._client = None
    
    def _get_client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel('gemini-1.5-pro')
        return self._client
    
    def infer(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Run inference via Gemini API.
        
        Args:
            model: Model name (ignored, uses gemini-1.5-pro)
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        import time
        
        client = self._get_client()
        if not client:
            return {
                'response': '',
                'latency_ms': 0,
                'model': 'gemini-1.5-pro',
                'success': False,
                'error': 'No API key configured'
            }
        
        start = time.perf_counter()
        try:
            response = client.generate_content(
                prompt,
                generation_config={
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_output_tokens': kwargs.get('max_tokens', 512),
                }
            )
            end = time.perf_counter()
            
            return {
                'response': response.text,
                'latency_ms': (end - start) * 1000,
                'model': 'gemini-1.5-pro',
                'success': True
            }
        except Exception as e:
            end = time.perf_counter()
            return {
                'response': '',
                'latency_ms': (end - start) * 1000,
                'model': 'gemini-1.5-pro',
                'success': False,
                'error': str(e)
            }
    
    def is_available(self) -> bool:
        """Check if Gemini API is configured."""
        return self.api_key is not None
