"""
Base router interface that all routing implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import time


class BaseRouter(ABC):
    """Abstract base class for all routers."""
    
    def __init__(self, model_pool=None):
        """
        Initialize the router.
        
        Args:
            model_pool: ModelPool instance containing available models
        """
        self.model_pool = model_pool
        self.routing_history = []
        
    @abstractmethod
    def route(self, query: str) -> str:
        """
        Route a query to the most appropriate model.
        
        Args:
            query: Input query string
            
        Returns:
            Model identifier (e.g., "gemini-1.5-pro", "llama-3-8b")
        """
        pass
    
    def route_with_metrics(self, query: str) -> Dict[str, Any]:
        """
        Route a query and return routing metrics.
        
        Args:
            query: Input query string
            
        Returns:
            Dictionary containing:
                - model: Selected model identifier
                - latency_ms: Routing latency in milliseconds
                - metadata: Additional routing metadata
        """
        start_time = time.perf_counter()
        model = self.route(query)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        result = {
            "model": model,
            "latency_ms": latency_ms,
            "query": query,
            "timestamp": time.time()
        }
        
        self.routing_history.append(result)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_history:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "model_distribution": {}
            }
        
        latencies = [h["latency_ms"] for h in self.routing_history]
        models = [h["model"] for h in self.routing_history]
        
        from collections import Counter
        model_counts = Counter(models)
        
        return {
            "total_queries": len(self.routing_history),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "model_distribution": dict(model_counts)
        }
