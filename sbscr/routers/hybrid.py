"""
Hybrid Router Implementation.
Combines SBSCR (Fast/Structural) with Semantic Router (Slow/Accurate).
"""

from typing import Dict, Any, Optional
from sbscr.routers.base import BaseRouter
from sbscr.routers.sbscr import SBSCRRouter
from sbscr.routers.semantic import SemanticRouter
from sbscr.core.models import ModelPool

class HybridRouter(BaseRouter):
    """
    Hybrid Router strategy:
    1. structural complexity < 4.0 -> SBSCR (Fastest)
    2. Code domain -> SBSCR (Fast & accurate for code)
    3. Ambiguous/Complex -> Semantic Router (High accuracy)
    """
    
    def __init__(self, model_pool: Optional[ModelPool] = None):
        super().__init__(model_pool or ModelPool())
        self.sbscr = SBSCRRouter(self.model_pool)
        self.semantic = SemanticRouter(self.model_pool)
        
    def route(self, query: str) -> str:
        """
        Smart Hybrid Routing:
        1. Run SBSCR (Fast).
        2. If SBSCR is 'confident' (Specific Model selected), use it.
        3. If SBSCR returns generic 'llama-3-8b', fallback to Semantic (Slow but Accurate).
        """
        # Step 1: Run Fast Router
        sbscr_model = self.sbscr.route(query)
        
        # Step 2: Confidence Check
        # SBSCR returns 'llama-3-8b' as a catch-all fallback. 
        # DeepSeek, Gemini, and Phi-3 are specific choices based on strong signals.
        
        if sbscr_model != 'llama-3-8b':
            return sbscr_model
            
        # Step 3: Semantic Fallback for Ambiguous Queries
        # If SBSCR wasn't sure (defaulted to Llama), let Semantic decide.
        # This pays the latency cost ONLY for ambiguous queries.
        return self.semantic.route(query)
    
    def route_with_explanation(self, query: str) -> Dict[str, Any]:
        sbscr_result = self.sbscr.route_with_explanation(query)
        model = sbscr_result['selected_model']
        
        if model != 'llama-3-8b':
            sbscr_result['strategy'] = 'Fast Path (Confident SBSCR)'
            return sbscr_result
            
        # Fallback
        semantic_result = self.semantic.route_with_explanation(query)
        semantic_result['strategy'] = 'Slow Path (Semantic Fallback)'
        return semantic_result

    def _is_obvious_code(self, query: str) -> bool:
        # Deprecated in favor of Smart Fallback
        return False

    def calibrate_from_examples(self, examples: list):
        pass
