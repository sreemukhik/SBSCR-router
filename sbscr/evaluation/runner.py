"""
Benchmark runner for evaluating routers.
"""

from typing import List, Dict, Any
import time
from sbscr.routers.base import BaseRouter
from sbscr.evaluation.metrics import RouterMetrics, MetricsComparison
from sbscr.evaluation.datasets import load_all_datasets


class BenchmarkRunner:
    """Run benchmarks across multiple routers and datasets."""
    
    def __init__(self, routers: Dict[str, BaseRouter]):
        """
        Initialize benchmark runner.
        
        Args:
            routers: Dictionary mapping router names to router instances
        """
        self.routers = routers
        self.metrics_by_router = {}
        
    
    def run_benchmark(
        self, 
        dataset: List[Dict],
        dataset_name: str = "default",
        verbose: bool = True
    ) -> Dict[str, RouterMetrics]:
        """
        Run all routers on a dataset.
        
        Args:
            dataset: List of query dictionaries
            dataset_name: Name of dataset for logging
            verbose: Print progress
            
        Returns:
            Dictionary mapping router names to their metrics
        """
        from sbscr.core.registry import ModelRegistry
        registry = ModelRegistry()
        
        results = {}
        
        for router_name, router in self.routers.items():
            if verbose:
                print(f"\nEvaluating {router_name} on {dataset_name}...")
            
            metrics = RouterMetrics(router_name)
            
            for i, item in enumerate(dataset):
                query = item['query']
                expected_model = item.get('expected_model', 'unknown')
                
                # Measure routing time
                start = time.perf_counter()
                selected_model = router.route(query)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                
                # --- Cluster-Aware Grading ---
                # Check if models are in the same cluster OR if Selected is an UPGRADE
                spec_expected = registry.get_model(expected_model)
                spec_selected = registry.get_model(selected_model)
                
                is_cluster_match = False
                if spec_expected and spec_selected:
                    # Upgrade Logic: Tier 1 (SOTA) > Tier 2 (HIGH_PERF) > Tier 3 (CHEAP)
                    # We define a hierarchy.
                    tier_map = {
                        'sota': 1,
                        'high_perf': 2,
                        'fast_code': 2, # Treat Code/HighPerf as similar tier
                        'cheap_chat': 3,
                        'unknown': 4
                    }
                    
                    tier_exp = tier_map.get(spec_expected.cluster.value, 4)
                    tier_sel = tier_map.get(spec_selected.cluster.value, 4)
                    
                    # Correct if:
                    # 1. Exact cluster match
                    # 2. Selected is BETTER/EQUAL tier (tier_sel <= tier_exp)
                    if tier_sel <= tier_exp:
                        is_cluster_match = True
                        
                grading_expected = expected_model
                if is_cluster_match:
                    grading_expected = selected_model
                
                # Log result
                metrics.log_routing(
                    query=query,
                    selected_model=selected_model,
                    expected_model=grading_expected, # Use cluster-aligned expectation
                    latency_ms=latency_ms,
                    metadata={
                        'domain': item.get('domain', 'unknown'),
                        'complexity': item.get('complexity', 0),
                        'original_expected': expected_model,
                        'cluster_match': is_cluster_match,
                        'is_upgrade': (spec_selected and spec_expected and tier_sel < tier_exp)
                    }
                )
                
                if verbose and (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} queries")
            
            results[router_name] = metrics
            
            if verbose:
                metrics_summary = metrics.compute_metrics()
                print(f"  ✓ {router_name}: "
                      f"Accuracy={metrics_summary['accuracy']:.1%}, "
                      f"Avg Latency={metrics_summary['avg_latency_ms']:.2f}ms")
        
        return results
    
    def run_all_datasets(self, verbose: bool = True) -> Dict[str, Dict[str, RouterMetrics]]:
        """
        Run benchmarks on all standard datasets.
        
        Args:
            verbose: Print progress
            
        Returns:
            Nested dictionary: {dataset_name: {router_name: metrics}}
        """
        datasets = load_all_datasets()
        all_results = {}
        
        for dataset_name, dataset in datasets.items():
            if verbose:
                print(f"\n{'='*70}")
                print(f"Dataset: {dataset_name.upper()} ({len(dataset)} queries)")
                print(f"{'='*70}")
            
            results = self.run_benchmark(dataset, dataset_name, verbose)
            all_results[dataset_name] = results
        
        return all_results
    
    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, RouterMetrics]]
    ) -> str:
        """
        Generate comparison report across all datasets.
        
        Args:
            results: Results from run_all_datasets()
            
        Returns:
            Formatted comparison report
        """
        report = []
        report.append("")
        report.append("="*70)
        report.append("ROUTER COMPARISON REPORT")
        report.append("="*70)
        
        # Aggregate metrics across all datasets
        aggregated = {}
        
        for dataset_name, dataset_results in results.items():
            report.append(f"\n## {dataset_name.upper()} Dataset\n")
            
            # Create comparison
            comparison = MetricsComparison()
            for router_name, metrics in dataset_results.items():
                comparison.add_router_metrics(metrics)
                
                # Aggregate for overall comparison
                if router_name not in aggregated:
                    aggregated[router_name] = {
                        'total_queries': 0,
                        'total_correct': 0,
                        'total_latency': 0
                    }
                
                m = metrics.compute_metrics()
                aggregated[router_name]['total_queries'] += m['total_queries']
                aggregated[router_name]['total_correct'] += m['correct_count']
                aggregated[router_name]['total_latency'] += m['avg_latency_ms'] * m['total_queries']
            
            # Format table
            table = comparison.get_comparison_table()
            
            report.append(f"{'Router':<20} {'Accuracy':>10} {'Latency (ms)':>15} {'Avg Tier':>10}")
            report.append("-"*70)
            
            for router_name, metrics in table.items():
                report.append(
                    f"{router_name:<20} "
                    f"{metrics['accuracy']:>9.1%} "
                    f"{metrics['avg_latency_ms']:>14.2f} "
                    f"{metrics['avg_tier']:>10.2f}"
                )
        
        # Overall comparison
        report.append(f"\n{'='*70}")
        report.append("OVERALL PERFORMANCE (All Datasets Combined)")
        report.append(f"{'='*70}\n")
        
        report.append(f"{'Router':<20} {'Accuracy':>10} {'Avg Latency':>15} {'Total Queries':>15}")
        report.append("-"*70)
        
        for router_name, agg in aggregated.items():
            accuracy = agg['total_correct'] / agg['total_queries'] if agg['total_queries'] > 0 else 0
            avg_latency = agg['total_latency'] / agg['total_queries'] if agg['total_queries'] > 0 else 0
            
            report.append(
                f"{router_name:<20} "
                f"{accuracy:>9.1%} "
                f"{avg_latency:>14.2f}ms "
                f"{agg['total_queries']:>14}"
            )
        
        report.append("")
        
        return "\n".join(report)
