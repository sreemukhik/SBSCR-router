from sbscr.routers.sbscr import SBSCRRouter
import json

def test_production_router():
    router = SBSCRRouter()
    
    test_cases = [
        "What is the capital of Japan?", # Simple
        "Write a React component that uses a custom hook to fetch data from a REST API and handles loading and error states." # Complex Coding
    ]
    
    print("\n🚀 --- ROUTER PRODUCTION TEST ---")
    for query in test_cases:
        print(f"\nQUERY: {query}")
        result = router.route_detailed(query)
        print(f"PATH  : {result['metrics'].get('routing_path')}")
        print(f"MODEL : {result['model']}")
        if "complexity_score" in result['metrics']:
            print(f"SCORE : {result['metrics']['complexity_score']:.4f}")

if __name__ == "__main__":
    test_production_router()
