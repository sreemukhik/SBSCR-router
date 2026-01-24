"""
Model Registry System.
Manages the database of available models, their capabilities, and dynamic clustering.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import yaml
import os
from enum import Enum

class ModelCluster(Enum):
    SOTA = "sota"           # Top-tier reasoning (GPT-4, Opus, Gemini 1.5)
    HIGH_PERF = "high_perf" # Strong open weights (Llama 3 70B, Mistral Large)
    FAST_CODE = "fast_code" # Optimized for code (DeepSeek V2, StarCoder)
    CHEAP_CHAT = "cheap_chat" # High speed, low cost (Llama 3 8b, Phi-3, Gemma)
    UNKNOWN = "unknown"

@dataclass
class ModelSpec:
    name: str
    provider: str
    cluster: ModelCluster
    context_window: int
    input_price_per_m: float # $ per 1M tokens
    output_price_per_m: float
    reasoning_score: float = 0.0 # MMLU or equivalent (0-100)
    coding_score: float = 0.0    # HumanEval (0-100)
    description: str = ""

class ModelRegistry:
    def __init__(self, config_path: str = "data/models.yaml"):
        self.config_path = config_path
        self.models: Dict[str, ModelSpec] = {}
        self.clusters: Dict[ModelCluster, List[str]] = {c: [] for c in ModelCluster}
        
        self.load_registry()

    def load_registry(self):
        """Load model definitions from YAML."""
        if not os.path.exists(self.config_path):
            print(f"WARN Registry not found at {self.config_path}. Using empty registry.")
            return

        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
            
        for name, spec in data.get('models', {}).items():
            cluster_str = spec.get('cluster', 'unknown').lower()
            try:
                cluster = ModelCluster(cluster_str)
            except ValueError:
                cluster = ModelCluster.UNKNOWN
                
            model = ModelSpec(
                name=name,
                provider=spec.get('provider', 'unknown'),
                cluster=cluster,
                context_window=spec.get('context_window', 4096),
                input_price_per_m=spec.get('price_in', 0.0),
                output_price_per_m=spec.get('price_out', 0.0),
                reasoning_score=spec.get('reasoning', 0.0),
                coding_score=spec.get('coding', 0.0),
                description=spec.get('description', '')
            )
            self.models[name] = model
            if cluster != ModelCluster.UNKNOWN:
                self.clusters[cluster].append(name)
                
    def get_candidates(self, cluster: ModelCluster, max_price: float = 100.0, min_context: int = 0) -> List[str]:
        """
        Get a list of candidate models in a cluster, sorted by price.
        Used for fallback chains.
        """
        candidates = [self.models[m] for m in self.clusters[cluster]]
        if not candidates:
            return []
            
        # 1. Filter by Context
        valid_candidates = [c for c in candidates if c.context_window >= min_context]
        candidates = valid_candidates if valid_candidates else []
        
        # 2. Filter by Price
        candidates = [c for c in candidates if c.input_price_per_m <= max_price]
        
        # 3. Sort by Price
        candidates.sort(key=lambda x: x.input_price_per_m)
        return [c.name for c in candidates]

    def get_best_model(self, cluster: ModelCluster, max_price: float = 100.0, min_context: int = 0) -> Optional[str]:
        """
        Find the best (cheapest/fastest) model in a cluster.
        Now filters by context window size (Constraint Layer).
        """
        candidates = self.get_candidates(cluster, max_price, min_context)
        return candidates[0] if candidates else None

    def get_model(self, model_name: str) -> Optional[ModelSpec]:
        """Retrieve a specific model specification."""
        return self.models.get(model_name)

    def list_models(self) -> List[str]:
        return list(self.models.keys())
