try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import time

class IntentClassifier:
    def __init__(self, model_name="valhalla/distilbart-mnli-12-1"):
        """
        Initializes the Zero-Shot Intent Classifier.
        Using distilbart-mnli-12-1 for speed/accuracy balance.
        """
        if not TRANSFORMERS_AVAILABLE:
            print("WARN Warning: transformers/torch not available. Intent classification will be disabled.")
            self.classifier = None
            return

        print(f"INFO Loading Intent Classifier: {model_name}...")
        start = time.time()
        
        # Check for CUDA
        device = 0 if torch.cuda.is_available() else -1
        self.device = device
        
        self.classifier = pipeline(
            "zero-shot-classification", 
            model=model_name, 
            device=device
        )
        self.labels = ["coding", "math", "creative", "reasoning", "general"]
        
        print(f"INFO Intent Classifier loaded in {time.time() - start:.2f}s (Device: {'GPU' if device==0 else 'CPU'})")

    def classify(self, text: str) -> str:
        """
        Classifies the intent of the text into one of the predefined labels.
        """
        if not self.classifier:
            return "general", 1.0

        start = time.time()
        result = self.classifier(text, self.labels)
        intent = result['labels'][0]
        score = result['scores'][0]
        
        # Debug logging
        # print(f"  🔍 Intent: {intent} ({score:.2f}) - {time.time() - start:.4f}s")
        
        return intent, score

if __name__ == "__main__":
    # Test
    classifier = IntentClassifier()
    queries = [
        "Write a Python function to sort a list.",
        "Solve for x: 2x + 5 = 10",
        "Write a poem about the autumn leaves.",
        "What is the capital of France?",
        "Explain the logic behind the Fermi paradox."
    ]
    
    print("\n--- Intent Classification Test ---")
    for q in queries:
        intent, score = classifier.classify(q)
        print(f"Query: '{q}'\n  -> Intent: {intent.upper()} ({score:.1%})\n")
