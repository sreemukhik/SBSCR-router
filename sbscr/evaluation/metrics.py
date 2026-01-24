"""
Metrics collection and analysis for router evaluation.
"""

from typing import Dict, List, Any
import time
from collections import defaultdict
import statistics


class RouterMetrics:
    """Collect and analyze routing metrics."""
    
    def __init__(self, router_name: str):
        """
        Initialize metrics collector.
        
        Args:
            router_name: Name of the router being evaluated
        """
        self.router_name = router_name
        self.results = []
        
    def log_routing(
        self, 
        query: str,
        selected_model: str,
        expected_model: str,
        latency_ms: float,
        metadata: Dict = None
    ):
        """
        Log a routing decision.
        
        Args:
            query: Input query
            selected_model: Model chosen by router
            expected_model: Ground truth optimal model
            latency_ms: Routing latency in milliseconds
            metadata: Additional metadata
        """
        result = {
            'query': query,
            'selected_model': selected_model,
            'expected_model': expected_model,
            'latency_ms': latency_ms,
            'correct': selected_model == expected_model,
            'metadata': metadata or {}
        }
        self.results.append(result)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.results:
            return {
                'router_name': self.router_name,
                'total_queries': 0,
                'accuracy': 0.0,
                'avg_latency_ms': 0.0
            }
        
        # Accuracy
        correct = sum(1 for r in self.results if r['correct'])
        accuracy = correct / len(self.results)
        
        # Latency stats
        latencies = [r['latency_ms'] for r in self.results]
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Model distribution
        model_counts = defaultdict(int)
        for r in self.results:
            model_counts[r['selected_model']] += 1
        
        # Tier distribution (for cost analysis)
        tier_map = {
            'phi-3-mini': 'tier_1_tiny',
            'llama-3-8b': 'tier_2_small',
            'deepseek-coder-6.7b': 'tier_3_medium',
            'gemini-1.5-pro': 'tier_4_large'
        }
        
        tier_counts = defaultdict(int)
        for model, count in model_counts.items():
            tier = tier_map.get(model, 'unknown')
            tier_counts[tier] += count
        
        return {
            'router_name': self.router_name,
            'total_queries': len(self.results),
            
            # Accuracy metrics
            'accuracy': accuracy,
            'correct_count': correct,
            'incorrect_count': len(self.results) - correct,
            
            # Latency metrics
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            
            # Distribution metrics
            'model_distribution': dict(model_counts),
            'tier_distribution': dict(tier_counts),
            
            # Cost efficiency (prefer lower tiers)
            'avg_tier': self._compute_avg_tier(model_counts),
        }
    
    def _compute_avg_tier(self, model_counts: Dict[str, int]) -> float:
        """Compute average tier used (lower is better for cost)."""
        tier_values = {
            'phi-3-mini': 1,
            'llama-3-8b': 2,
            'deepseek-coder-6.7b': 3,
            'gemini-1.5-pro': 4
        }
        
        total = 0
        count = 0
        for model, freq in model_counts.items():
            tier = tier_values.get(model, 2.5)
            total += tier * freq
            count += freq
        
        return total / count if count > 0 else 0
    
    def get_errors(self) -> List[Dict]:
        """Get all incorrect routing decisions for analysis."""
        return [r for r in self.results if not r['correct']]
    
    def to_dict(self) -> Dict:
        """Export all results as dictionary."""
        return {
            'router_name': self.router_name,
            'metrics': self.compute_metrics(),
            'results': self.results
        }


class MetricsComparison:
    """Compare metrics across multiple routers."""
    
    def __init__(self):
        """Initialize comparison tool."""
        self.router_metrics = {}
        
    def add_router_metrics(self, metrics: RouterMetrics):
        """Add metrics from a router."""
        self.router_metrics[metrics.router_name] = metrics.compute_metrics()
    
    def get_comparison_table(self) -> Dict[str, Dict]:
        """Get comparison table of all routers."""
        return self.router_metrics
    
    def get_best_router(self, metric: str = 'accuracy') -> str:
        """
        Get best router by a specific metric.
        
        Args:
            metric: Metric to optimize ('accuracy', 'avg_latency_ms', etc.)
            
        Returns:
            Name of best router
        """
        if not self.router_metrics:
            return None
        
        # For latency, lower is better
        if 'latency' in metric:
            return min(self.router_metrics.items(), key=lambda x: x[1][metric])[0]
        else:
            # For other metrics, higher is better
            return max(self.router_metrics.items(), key=lambda x: x[1][metric])[0]
