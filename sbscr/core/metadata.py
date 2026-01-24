"""
Structural complexity metadata extractor.
Analyzes queries to estimate task difficulty without running inference.
"""

from typing import Dict, Any
import re
import math
from collections import Counter


class ComplexityExtractor:
    """Extract structural features from queries to estimate complexity."""
    
    def __init__(self):
        """Initialize complexity extractor with feature weights."""
        # Complexity indicators (keywords that suggest difficulty)
        self.complexity_keywords = {
            # High complexity
            'implement', 'design', 'architect', 'optimize', 'analyze',
            'complex', 'advanced', 'detailed', 'comprehensive', 'sophisticated',
            'algorithm', 'distributed', 'concurrent', 'parallel', 'async',
            'recursive', 'dynamic programming', 'graph', 'tree', 'optimization',
            'prove', 'proof', 'undecidable', 'theorem', 'lemma', 'formal',
            
            # Medium complexity  
            'create', 'build', 'develop', 'write', 'generate',
            'explain', 'describe', 'compare', 'contrast',
            
            # Low complexity
            'what', 'define', 'list', 'simple', 'basic', 'quick',
            'hello world', 'example', 'print'
        }
        
        self.complexity_weights = {
            'implement': 3, 'design': 3, 'architect': 3, 'optimize': 3,
            'complex': 3, 'advanced': 3, 'comprehensive': 3, 'sophisticated': 3,
            'algorithm': 2.5, 'distributed': 3, 'concurrent': 2.5, 'recursive': 2,
            'prove': 3.5, 'proof': 3.5, 'undecidable': 4, 'theorem': 3,
            'lemma': 3, 'formal': 2.5, 'reduction': 2.5,
            'consensus': 3, 'consistency': 3, 'compiler': 3, 'parser': 3, 
            'interpreter': 3, 'neural network': 3, 'training': 2.5,
            'create': 1.5, 'build': 1.5, 'explain': 1, 'describe': 1,
            'what': 0.5, 'simple': 0.3, 'basic': 0.3, 'quick': 0.5,
            'hello world': 0.1, 'print': 0.3
        }
        
        # Domain markers
        self.domain_markers = {
            'code': ['function', 'class', 'def', 'return', 'import', '()', '{}', 'async', 'await', 'compiler', 'parser', 'consensus'],
            'math': ['equation', 'solve', 'calculate', 'derivative', 'integral', '∫', '∑', 'lim'],
            'reasoning': ['therefore', 'because', 'assume', 'prove', 'logic', 'if-then', 'axiom'],
            'creative': ['story', 'poem', 'creative', 'imagine', 'generate'],
        }
        
    def extract_features(self, query: str) -> Dict[str, Any]:
        """
        Extract all structural features from a query.
        
        Args:
            query: Input query text
            
        Returns:
            Dictionary of feature names and values
        """
        query_lower = query.lower()
        
        features = {
            # Length features
            'char_count': len(query),
            'word_count': len(query.split()),
            'avg_word_length': self._avg_word_length(query),
            
            # Syntactic features
            'sentence_count': len(re.split(r'[.!?]+', query)),
            'question_marks': query.count('?'),
            'code_blocks': len(re.findall(r'```.*?```', query, re.DOTALL)),
            'parentheses_depth': self._max_nesting_depth(query, '(', ')'),
            'bracket_depth': self._max_nesting_depth(query, '[', ']'),
            'brace_depth': self._max_nesting_depth(query, '{', '}'),
            
            # Lexical complexity
            'unique_word_ratio': self._unique_word_ratio(query),
            'rare_word_ratio': self._rare_word_ratio(query),
            
            # Domain detection
            'domain': self._detect_domain(query_lower),
            'is_code_related': self._is_code_related(query),
            'is_math_related': self._is_math_related(query),
            
            # Advanced Code Features
            **self._analyze_code_complexity(query),
            
            # Complexity indicators
            'complexity_keyword_count': self._count_complexity_keywords(query_lower),
            'complexity_keyword_score': self._score_complexity_keywords(query_lower),
            'has_multi_step': self._has_multi_step_indicators(query_lower),
            'has_constraints': self._has_constraints(query_lower),
        }
        
        return features
    
    def estimate_complexity(self, query: str) -> float:
        """
        Estimate overall complexity score (0-10 scale).
        
        Args:
            query: Input query text
            
        Returns:
            Complexity score from 0 (trivial) to 10 (very complex)
        """
        features = self.extract_features(query)
        
        # Check for simple patterns first (fast path)
        if self._is_simple_pattern(query):
            return min(features['complexity_keyword_score'] + 0.5, 2.0)
        
        score = 0.0
        
        # Length contribution (longer queries often more complex)
        word_count = features['word_count']
        if word_count < 5:
            score += 0.5
        elif word_count < 15:
            score += 2.0
        elif word_count < 30:
            score += 4.0
        else:
            score += 6.0
        
        # Keyword-based complexity
        # Increased cap from 4.0 to 6.0 to capture highly technical queries
        score += min(features['complexity_keyword_score'], 6.0)
        
        # Structural complexity
        if features['code_blocks'] > 0:
            score += 2.0
        if features['parentheses_depth'] > 2:
            score += 1.0
        if features['has_multi_step']:
            score += 1.5
        if features['has_constraints']:
            score += 1.0
            
        # Domain-specific adjustments (NEW)
        domain = features['domain']
        query_lower = query.lower()
        
        # Code domain boosting
        if features['is_code_related']:
            # Algorithmic/system design keywords
            if any(kw in query_lower for kw in ['algorithm', 'implement', 'design', 'architecture']):
                score += 2.0  # Boosted from 1.5
            # Concurrent/distributed keywords
            if any(kw in query_lower for kw in ['distributed', 'concurrent', 'parallel', 'thread']):
                score += 2.5  # Boosted from 2.0
        
        # Math/reasoning domain boosting
        if features['is_math_related']:
            # Theoretical math keywords
            if any(kw in query_lower for kw in ['prove', 'theorem', 'derive', 'formal']):
                score += 2.5  # Boosted from 2.0
            # Advanced topics
            if any(kw in query_lower for kw in ['integral', 'derivative', 'matrix', 'differential']):
                score += 1.0
        
        # Normalize to 0-10 scale
        score = min(max(score, 0), 10)
        
        return round(score, 2)
    
    def _is_simple_pattern(self, query: str) -> bool:
        """Check if query matches simple patterns (for fast-path routing)."""
        import re
        query_lower = query.lower().strip()
        
        # Very short queries
        if len(query_lower) < 15 and '?' in query_lower:
            return True
        
        # Simple arithmetic
        if re.match(r'^(what is |calculate )?\d+\s*[\+\-\*/]\s*\d+', query_lower):
            return True
        
        # Definition queries
        if query_lower.startswith(('what is ', 'define ', 'what are ')):
            return True
        
        # Hello world / basic examples
        if any(p in query_lower for p in ['hello world', 'print hello', 'basic example']):
            return True
        
        return False
    
    # Helper methods
    
    def _avg_word_length(self, query: str) -> float:
        """Calculate average word length."""
        words = query.split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    def _max_nesting_depth(self, text: str, open_char: str, close_char: str) -> int:
        """Calculate maximum nesting depth of brackets/parentheses."""
        depth = 0
        max_depth = 0
        for char in text:
            if char == open_char:
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == close_char:
                depth = max(0, depth - 1)
        return max_depth
    
    def _unique_word_ratio(self, query: str) -> float:
        """Calculate ratio of unique words to total words."""
        words = query.lower().split()
        if not words:
            return 0
        return len(set(words)) / len(words)
    
    def _rare_word_ratio(self, query: str) -> float:
        """Estimate ratio of rare/technical words (simple heuristic)."""
        words = query.split()
        if not words:
            return 0
        # Words longer than 10 chars are often technical/rare
        long_words = sum(1 for word in words if len(word) > 10)
        return long_words / len(words)
    
    def _detect_domain(self, query: str) -> str:
        """Detect primary domain of the query."""
        domain_scores = {}
        for domain, markers in self.domain_markers.items():
            score = sum(1 for marker in markers if marker in query)
            domain_scores[domain] = score
        
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    def _is_code_related(self, query: str) -> bool:
        """Check if query is code-related."""
        code_indicators = ['function', 'code', 'program', 'script', 'implement',
                          'class', 'method', 'algorithm', 'def ', 'return ']
        return any(indicator in query.lower() for indicator in code_indicators)
    
    def _is_math_related(self, query: str) -> bool:
        """Check if query is math-related."""
        math_indicators = ['calculate', 'solve', 'equation', 'formula', 'derivative',
                          'integral', 'proof', 'theorem', 'mathematical']
        return any(indicator in query.lower() for indicator in math_indicators)
    
    def _count_complexity_keywords(self, query: str) -> int:
        """Count number of complexity-indicating keywords."""
        return sum(1 for keyword in self.complexity_keywords if keyword in query)
    
    def _score_complexity_keywords(self, query: str) -> float:
        """Score complexity based on weighted keywords."""
        score = 0
        for keyword, weight in self.complexity_weights.items():
            if keyword in query:
                score += weight
        return score
    
    def _has_multi_step_indicators(self, query: str) -> bool:
        """Check if query suggests multi-step process."""
        multi_step_words = ['step', 'first', 'then', 'finally', 'next', 'also', 'additionally']
        return sum(1 for word in multi_step_words if word in query) >= 2
    
    def _has_constraints(self, query: str) -> bool:
        """Check if query has explicit constraints."""
        constraint_words = ['without', 'must', 'should', 'requirement', 'constraint',
                           'only', 'except', 'but not', 'while']
        return any(word in query for word in constraint_words)

    # --- Advanced Code Analysis (New for Big Data) ---
    
    def _analyze_code_complexity(self, query: str) -> Dict[str, float]:
        """
        Analyze code blocks using AST (if Python) or heuristics.
        Returns dictionary of code-specific features.
        """
        import ast
        
        # Extract potential python code blocks
        code_blocks = re.findall(r'```(?:python)?(.*?)```', query, re.DOTALL)
        
        # Heuristic: If explicitly python code block, prioritize it.
        # If not, but query is code-heavy, try strictly parsing it only if it looks passable.
        if not code_blocks:
            if self._is_code_related(query) and ('def ' in query or 'import ' in query or '=' in query):
                code_blocks = [query]
            else:
                return {'code_density': 0, 'ast_depth': 0, 'import_count': 0}
                
        max_depth = 0
        total_imports = 0
        
        for block in code_blocks:
            try:
                tree = ast.parse(block)
                
                # 1. Calculate true AST depth
                class DepthVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.max_depth = 0
                        self.current_depth = 0
                    
                    def generic_visit(self, node):
                        self.current_depth += 1
                        self.max_depth = max(self.max_depth, self.current_depth)
                        super().generic_visit(node)
                        self.current_depth -= 1
                
                visitor = DepthVisitor()
                visitor.visit(tree)
                max_depth = max(max_depth, visitor.max_depth)
                
                # 2. Count Imports
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        total_imports += 1
                        
            except SyntaxError:
                continue
                
        # Normalize features
        # Depth > 10 is very complex. Imports > 5 is system-level.
        return {
            'code_density': len("".join(code_blocks)) / (len(query) + 1),
            'ast_depth': float(max_depth),
            'import_count': float(total_imports)
        }
