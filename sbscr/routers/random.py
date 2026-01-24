"""
Random Router - Control Baseline
Randomly selects models from the pool without any intelligence.
"""

import random
from typing import Optional
from sbscr.routers.base import BaseRouter
from sbscr.core.models import ModelPool


class RandomRouter(BaseRouter):
    """
    Random routing baseline.
    Establishes performance floor - what happens with no routing intelligence.
    """
    
    def __init__(self, model_pool: Optional[ModelPool] = None, seed: Optional[int] = None):
        """
        Initialize random router.
        
        Args:
            model_pool: ModelPool instance (creates default if None)
            seed: Random seed for reproducibility (None = non-deterministic)
        """
        super().__init__(model_pool or ModelPool())
        
        if seed is not None:
            random.seed(seed)
            self._seed = seed
        else:
            self._seed = None
            
        self.available_models = self.model_pool.get_all_models()
        
    def route(self, query: str) -> str:
        """
        Randomly select a model from the pool.
        
        Args:
            query: Input query (ignored)
            
        Returns:
            Randomly selected model name
        """
        return random.choice(self.available_models)
    
    def get_stats(self):
        """Get routing statistics with distribution check."""
        stats = super().get_stats()
        
        # Add expected distribution for comparison
        if stats['total_queries'] > 0:
            expected_per_model = stats['total_queries'] / len(self.available_models)
            stats['expected_per_model'] = expected_per_model
            
            # Check if distribution is roughly uniform
            actual_counts = stats['model_distribution'].values()
            max_deviation = max(abs(count - expected_per_model) for count in actual_counts)
            stats['max_deviation_from_uniform'] = max_deviation
            
        return stats
