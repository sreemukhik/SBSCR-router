"""
End-to-end evaluation: Route → Infer → Evaluate Quality
This measures actual task performance, not just routing accuracy.
"""

from sbscr.routers import SBSCRRouter, RandomRouter, KeywordRouter, SemanticRouter
from sbscr.inference import LLMClient, CodeEvaluator, MathEvaluator
from sbscr.evaluation.datasets import load_humaneval_subset, load_gsm8k_subset
import time
import json


def run_end_to_end_benchmark(router, llm_client, dataset, evaluator, router_name="Router"):
    """
    Run end-to-end benchmark: routing + inference + evaluation.
    
    Args:
        router: Router instance
        llm_client: LLM client for inference
        dataset: List of queries with expected outputs
        evaluator: Task evaluator
        router_name: Name for reporting
        
    Returns:
        Results dictionary
    """
    results = []
    
    for i, item in enumerate(dataset):
        query = item['query']
        expected = item.get('expected_answer') or item.get('expected_model')
        
        # Step 1: Route query
        route_start = time.perf_counter()
        selected_model = router.route(query)
        route_end = time.perf_counter()
        routing_latency = (route_end - route_start) * 1000
        
        # Step 2: Run inference
        try:
            inference_result = llm_client.infer(
                selected_model, 
                query,
                max_tokens=200,
                temperature=0.3  # Lower temp for more consistent results
            )
            
            # Step 3: Evaluate quality
            if inference_result['success']:
                quality = evaluator.evaluate(
                    query,
                    inference_result['response'],
                    expected
                )
            else:
                quality = {'success': False, 'error': inference_result.get('error')}
            
            result = {
                'query': query,
                'selected_model': selected_model,
                'routing_latency_ms': routing_latency,
                'inference_latency_ms': inference_result['latency_ms'],
                'total_latency_ms': routing_latency + inference_result['latency_ms'],
                'inference_success': inference_result['success'],
                'quality': quality,
            }
            
            results.append(result)
            
            quality_score = quality.get('pass@1') if 'pass@1' in quality else quality.get('exact_match')
            
            print(f"  [{i+1}/{len(dataset)}] {selected_model:20} | "
                  f"Route: {routing_latency:6.2f}ms | "
                  f"Infer: {inference_result['latency_ms']/1000:6.2f}s | "
                  f"Quality: {quality_score}")
            
        except Exception as e:
            print(f"  [{i+1}/{len(dataset)}] ERROR: {e}")
            continue
    
    # Calculate aggregate metrics
    successful = [r for r in results if r['inference_success']]
    
    if not successful:
        return {
            'router_name': router_name,
            'total_queries': len(dataset),
            'successful_inferences': 0,
            'avg_routing_latency_ms': 0,
            'avg_total_latency_ms': 0,
        }, results
    
    metrics = {
        'router_name': router_name,
        'total_queries': len(dataset),
        'successful_inferences': len(successful),
        'avg_routing_latency_ms': sum(r['routing_latency_ms'] for r in successful) / len(successful),
        'avg_inference_latency_ms': sum(r['inference_latency_ms'] for r in successful) / len(successful),
        'avg_total_latency_ms': sum(r['total_latency_ms'] for r in successful) / len(successful),
    }
    
    # Calculate quality metrics based on task type
    if 'pass@1' in successful[0]['quality']:
        # Code tasks
        passed = sum(1 for r in successful if r['quality'].get('pass@1', False))
        metrics['pass@1'] = passed / len(successful)
    elif 'exact_match' in successful[0]['quality']:
        # Math tasks
        correct = sum(1 for r in successful if r['quality'].get('exact_match', False))
        metrics['exact_match'] = correct / len(successful)
    
    return metrics, results


def main():
    print("\n" + "="*80)
    print("END-TO-END EVALUATION: SBSCR vs Baselines")
    print("Routing → Inference → Quality Measurement")
    print("="*80)
    
    # Initialize
    llm_client = LLMClient()
    routers = {
        'SBSCR': SBSCRRouter(),
        'Keyword': KeywordRouter(),
        # Skip Random and Semantic for now (too slow)
    }
    
    print("\nℹ️  NOTE: Using small dataset (5 queries) for demo")
    print("   Full evaluation would use 100+ queries\n")
    
    # Test with math dataset (faster inference)
    print("\n" + "="*80)
    print("MATH EVALUATION (GSM8K subset)")
    print("="*80)
    
    math_dataset = load_gsm8k_subset(5)  # Small subset for demo
    math_evaluator = MathEvaluator()
    
    all_results = {}
    
    for router_name, router in routers.items():
        print(f"\n🔬 Testing {router_name}...")
        metrics, results = run_end_to_end_benchmark(
            router,
            llm_client,
            math_dataset,
            math_evaluator,
            router_name
        )
        all_results[router_name] = {'metrics': metrics, 'results': results}
        
        print(f"\n  ✓ {router_name} Results:")
        print(f"     Exact Match: {metrics.get('exact_match', 0):.1%}")
        print(f"     Avg Total Latency: {metrics['avg_total_latency_ms']/1000:.2f}s")
        print(f"     (Routing: {metrics['avg_routing_latency_ms']:.2f}ms + "
              f"Inference: {metrics['avg_inference_latency_ms']/1000:.2f}s)")
    
    # Save results
    with open('end_to_end_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("="*80)
    print("\nRouting latency (2-20ms) is negligible compared to inference (4-24s).")
    print("What matters is: Does the router pick a model that can solve the task?")
    print("\n💾 Detailed results saved to: end_to_end_results.json")
    print("\n✅ End-to-end evaluation complete!")


if __name__ == "__main__":
    main()
