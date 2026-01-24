"""
Semantic Router - Embedding-based routing baseline.
Uses sentence embeddings for semantic similarity matching.
"""

from typing import Optional, Dict, List
import numpy as np
from sbscr.routers.base import BaseRouter
from sbscr.core.models import ModelPool


class SemanticRouter(BaseRouter):
    """
    Semantic routing using sentence embeddings.
    Routes based on semantic similarity to category prototypes.
    """
    
    def __init__(self, model_pool: Optional[ModelPool] = None):
        """
        Initialize semantic router.
        
        Args:
            model_pool: ModelPool instance (creates default if None)
        """
        super().__init__(model_pool or ModelPool())
        
        # Lazy load sentence transformer to avoid slow import
        self._encoder = None
        
        # Define category prototypes (example queries for each model tier)
        self.category_prototypes = {
            'phi-3-mini': [
                "What is machine learning?",
                "Define a variable",
                "Print hello world",
                "What is 2 + 2?",
                "Explain recursion briefly"
            ],
            'llama-3-8b': [
                "Write a function to reverse a string",
                "Explain how binary search works",
                "Calculate the derivative of x^2",
                "Compare Python and JavaScript",
                "Implement bubble sort"
            ],
            'deepseek-coder-6.7b': [
                "Implement a binary search tree",
                "Write a parser for JSON",
                "Create a web scraper",
                "Build a REST API endpoint",
                "Implement quicksort algorithm"
            ],
            'gemini-1.5-pro': [
                "Design a distributed consensus algorithm",
                "Implement a compiler for a language",
                "Prove the halting problem is undecidable",
                "Design a scalable microservices architecture",
                "Optimize a complex database query"
            ]
        }
        
        # Pre-computed prototype embeddings (computed on first use)
        self._prototype_embeddings = None
        
    def _get_encoder(self):
        """Lazy load sentence transformer model."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            # Use lightweight model for speed
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._encoder
    
    def _compute_prototype_embeddings(self):
        """Compute embeddings for all category prototypes."""
        if self._prototype_embeddings is not None:
            return
        
        encoder = self._get_encoder()
        self._prototype_embeddings = {}
        
        for model, examples in self.category_prototypes.items():
            # Compute embedding for each example
            embeddings = encoder.encode(examples)
            # Average embeddings to get category centroid
            centroid = np.mean(embeddings, axis=0)
            self._prototype_embeddings[model] = centroid
    
    def route(self, query: str) -> str:
        """
        Route query based on semantic similarity to prototypes.
        
        Args:
            query: Input query
            
        Returns:
            Selected model name
        """
        # Ensure prototypes are computed
        self._compute_prototype_embeddings()
        
        encoder = self._get_encoder()
        
        # Encode query
        query_embedding = encoder.encode([query])[0]
        
        # Compute cosine similarity to each prototype
        similarities = {}
        for model, prototype_emb in self._prototype_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, prototype_emb)
            similarities[model] = similarity
        
        # Return model with highest similarity
        return max(similarities, key=similarities.get)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def route_with_explanation(self, query: str) -> Dict:
        """Route with similarity scores for all models."""
        self._compute_prototype_embeddings()
        encoder = self._get_encoder()
        
        query_embedding = encoder.encode([query])[0]
        
        similarities = {}
        for model, prototype_emb in self._prototype_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, prototype_emb)
            similarities[model] = float(similarity)
        
        selected_model = max(similarities, key=similarities.get)
        
        return {
            'selected_model': selected_model,
            'similarities': similarities,
            'reasoning': f"Highest similarity to '{selected_model}' prototypes ({similarities[selected_model]:.3f})"
        }
