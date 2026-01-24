"""
Keyword-based Heuristic Router
Fast rule-based routing using domain keywords and complexity indicators.
"""

from typing import Optional, Dict, List
from sbscr.routers.base import BaseRouter
from sbscr.core.models import ModelPool


class KeywordRouter(BaseRouter):
    """
    Simple rule-based router using keyword matching.
    Fast but limited - no structural complexity analysis.
    """
    
    def __init__(self, model_pool: Optional[ModelPool] = None):
        """
        Initialize keyword router.
        
        Args:
            model_pool: ModelPool instance (creates default if None)
        """
        super().__init__(model_pool or ModelPool())
        
        # Domain detection patterns
        self.domain_patterns = {
            'code': [
                'function', 'code', 'program', 'implement', 'algorithm',
                'class', 'method', 'script', 'debug', 'compile',
                'python', 'javascript', 'java', 'c++', 'rust',
                'def ', 'return', 'import', 'for loop', 'if statement'
            ],
            'math': [
                'calculate', 'solve', 'equation', 'formula', 'derivative',
                'integral', 'matrix', 'algebra', 'calculus', 'geometry',
                'prove', 'theorem', 'mathematical', 'compute'
            ],
            'reasoning': [
                'explain', 'why', 'analyze', 'compare', 'reasoning',
                'logic', 'argument', 'conclude', 'infer', 'deduce'
            ],
            'creative': [
                'story', 'poem', 'creative', 'write', 'imagine',
                'generate', 'compose', 'draft', 'narrative'
            ]
        }
        
        # Complexity indicators (high complexity keywords)
        self.complexity_keywords = [
            'complex', 'advanced', 'sophisticated', 'comprehensive',
            'distributed', 'concurrent', 'parallel', 'optimize',
            'design', 'architect', 'system', 'scalable',
            'prove', 'formal', 'theorem', 'undecidable'
        ]
        
        # Simple length-based complexity (rough heuristic)
        self.length_thresholds = {
            'short': 10,    # < 10 words = simple
            'medium': 20,   # 10-20 words = medium
            'long': 20      # > 20 words = complex
        }
        
    def route(self, query: str) -> str:
        """
        Route query based on keyword matching.
        
        Args:
            query: Input query
            
        Returns:
            Selected model name
        """
        query_lower = query.lower()
        
        # Detect domain
        domain = self._detect_domain(query_lower)
        
        # Estimate complexity (simple heuristic)
        is_complex = self._is_complex(query_lower)
        
        # Routing logic
        word_count = len(query.split())
        
        # Very complex queries → Gemini
        if is_complex or any(kw in query_lower for kw in ['prove', 'theorem', 'distributed', 'concurrent']):
            return 'gemini-1.5-pro'
        
        # Code domain
        if domain == 'code':
            # Long code queries → specialized coder
            if word_count > 15:
                return 'deepseek-coder-6.7b'
            # Medium code queries → general small model
            elif word_count > 8:
                return 'llama-3-8b'
            # Simple code → tiny model
            else:
                return 'phi-3-mini'
        
        # Math domain
        if domain == 'math':
            if is_complex:
                return 'gemini-1.5-pro'
            else:
                return 'llama-3-8b'
        
        # Reasoning/creative → medium to large
        if domain in ['reasoning', 'creative']:
            if is_complex:
                return 'gemini-1.5-pro'
            else:
                return 'llama-3-8b'
        
        # Default: general queries
        if word_count > 15:
            return 'llama-3-8b'
        else:
            return 'phi-3-mini'
    
    def _detect_domain(self, query: str) -> str:
        """Detect query domain using keyword matching."""
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for kw in keywords if kw in query)
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no matches
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    def _is_complex(self, query: str) -> bool:
        """Check if query contains complexity indicators."""
        return any(kw in query for kw in self.complexity_keywords)
    
    def route_with_explanation(self, query: str) -> Dict:
        """Route with detailed explanation of reasoning."""
        query_lower = query.lower()
        domain = self._detect_domain(query_lower)
        is_complex = self._is_complex(query_lower)
        word_count = len(query.split())
        
        model = self.route(query)
        
        return {
            'selected_model': model,
            'domain': domain,
            'is_complex': is_complex,
            'word_count': word_count,
            'reasoning': f"Domain: {domain}, Complex: {is_complex}, Words: {word_count}"
        }
