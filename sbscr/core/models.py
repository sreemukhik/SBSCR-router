"""
Model pool management for routing decisions.
Defines model capabilities and handles model selection logic.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ModelTier(Enum):
    """Model capability tiers."""
    TINY = 1      # Trivial tasks only
    SMALL = 2     # Simple tasks
    MEDIUM = 3    # Medium complexity
    LARGE = 4     # Complex tasks
    

@dataclass
class ModelCapability:
    """Defines a model's capabilities and constraints."""
    
    name: str
    tier: ModelTier
    domains: List[str]  # e.g., ['code', 'math', 'reasoning']
    complexity_threshold: float  # Max complexity score (0-10)
    cost_per_1k_tokens: float = 0.0
    latency_ms: float = 0.0  # Average inference latency
    max_tokens: int = 4096
    
    def can_handle(self, domain: str, complexity: float) -> bool:
        """Check if this model can handle a given task."""
        domain_match = 'general' in self.domains or domain in self.domains
        complexity_ok = complexity <= self.complexity_threshold
        return domain_match and complexity_ok
    
    def __repr__(self):
        return f"Model({self.name}, tier={self.tier.name}, max_complexity={self.complexity_threshold})"


class ModelPool:
    """Manages available models and their capabilities."""
    
    def __init__(self):
        """Initialize model pool with default configuration."""
        self.models: Dict[str, ModelCapability] = {}
        self._load_default_models()
        
    def _load_default_models(self):
        """Load default model configurations."""
        
        # Tier 1: Tiny - Simple queries (EXPANDED from 2.0 to 3.0)
        self.add_model(ModelCapability(
            name="phi-3-mini",
            tier=ModelTier.TINY,
            domains=['general', 'code'],
            complexity_threshold=3.0,  # ✅ Increased to capture more simple queries
            cost_per_1k_tokens=0.0,  # Local
            latency_ms=100,
            max_tokens=4096
        ))
        
        # Tier 2: Small - Easy to medium tasks
        self.add_model(ModelCapability(
            name="llama-3-8b",
            tier=ModelTier.SMALL,
            domains=['general', 'code', 'reasoning'],
            complexity_threshold=6.0,  # Adjusted from 5.0 to 6.0
            cost_per_1k_tokens=0.0,  # Groq free tier
            latency_ms=500,
            max_tokens=8192
        ))
        
        # Tier 3: Medium - Specialized code tasks (LOWERED from 7.0 to 5.0)
        self.add_model(ModelCapability(
            name="deepseek-coder-6.7b",
            tier=ModelTier.MEDIUM,
            domains=['code'],
            complexity_threshold=8.0,  # ✅ Lowered to actually get used
            cost_per_1k_tokens=0.0,  # Local
            latency_ms=800,
            max_tokens=4096
        ))
        
        # Tier 4: Large - Complex tasks (ADDED threshold instead of "all")
        self.add_model(ModelCapability(
            name="gemini-1.5-pro",
            tier=ModelTier.LARGE,
            domains=['general', 'code', 'math', 'reasoning', 'creative'],
            complexity_threshold=10.0,  # ✅ Reserve for truly complex (7.0+)
            cost_per_1k_tokens=0.0,  # Student free tier
            latency_ms=2000,
            max_tokens=1000000
        ))
        
    def add_model(self, model: ModelCapability):
        """Add a model to the pool."""
        self.models[model.name] = model
        
    def remove_model(self, model_name: str):
        """Remove a model from the pool."""
        if model_name in self.models:
            del self.models[model_name]
            
    def get_model(self, model_name: str) -> Optional[ModelCapability]:
        """Get model by name."""
        return self.models.get(model_name)
    
    def get_capable_models(self, domain: str, complexity: float) -> List[ModelCapability]:
        """
        Get all models capable of handling a task.
        
        Args:
            domain: Task domain (e.g., 'code', 'math')
            complexity: Complexity score (0-10)
            
        Returns:
            List of capable models, sorted by tier (smallest to largest)
        """
        capable = [
            model for model in self.models.values()
            if model.can_handle(domain, complexity)
        ]
        return sorted(capable, key=lambda m: m.tier.value)
    
    def select_optimal_model(
        self, 
        domain: str, 
        complexity: float,
        prefer_cost: bool = True
    ) -> Optional[str]:
        """
        Select the optimal model for a task.
        
        Args:
            domain: Task domain
            complexity: Complexity score
            prefer_cost: If True, prefer cheaper models when capable
            
        Returns:
            Model name, or None if no capable model found
        """
        capable = self.get_capable_models(domain, complexity)
        
        if not capable:
            # Fallback to most powerful model
            return "gemini-1.5-pro"
        
        if prefer_cost:
            # Return smallest capable model (best cost efficiency)
            return capable[0].name
        else:
            # Return most powerful capable model
            return capable[-1].name
    
    def get_all_models(self) -> List[str]:
        """Get list of all model names."""
        return list(self.models.keys())
    
    def __repr__(self):
        return f"ModelPool({len(self.models)} models: {', '.join(self.models.keys())})"
