"""
Task evaluators for measuring output quality.
"""

import re
import sys
import io
import contextlib
from typing import Dict, Any, Optional


class TaskEvaluator:
    """Base class for task evaluation."""
    
    def evaluate(self, query: str, response: str, expected: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate response quality.
        
        Args:
            query: Original query
            response: Model response
            expected: Expected/reference answer (if available)
            
        Returns:
            Evaluation metrics
        """
        raise NotImplementedError


class CodeEvaluator(TaskEvaluator):
    """Evaluator for code generation tasks."""
    
    def evaluate(self, query: str, response: str, test_cases: Optional[list] = None) -> Dict[str, Any]:
        """
        Evaluate generated code.
        
        Args:
            query: Original query
            response: Generated code
            test_cases: List of (input, expected_output) tuples
            
        Returns:
            Dictionary with 'pass@1', 'executable', 'error'
        """
        # Extract code from response
        code = self._extract_code(response)
        
        if not code:
            return {
                'pass@1': False,
                'executable': False,
                'error': 'No code found in response',
                'code': None
            }
        
        # Test if code is executable
        executable, error = self._test_execution(code)
        
        result = {
            'executable': executable,
            'code': code,
            'error': error
        }
        
        # If test cases provided, check correctness
        if test_cases and executable:
            passed = self._run_test_cases(code, test_cases)
            result['pass@1'] = passed
        else:
            result['pass@1'] = executable  # At least it runs
        
        return result
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from markdown or plain text."""
        # Try to find code in markdown blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no markdown blocks, look for function/class definitions
        if 'def ' in response or 'class ' in response:
            # Extract lines that look like code
            lines = []
            in_code = False
            for line in response.split('\n'):
                if 'def ' in line or 'class ' in line:
                    in_code = True
                if in_code:
                    lines.append(line)
                if in_code and line and not line[0].isspace() and line.strip():
                    if not ('def ' in line or 'class ' in line or line.strip().startswith('#')):
                        break
            
            return '\n'.join(lines).strip()
        
        return None
    
    def _test_execution(self, code: str) -> tuple:
        """Test if code executes without errors."""
        try:
            # Create isolated namespace
            namespace = {}
            exec(code, namespace)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def _run_test_cases(self, code: str, test_cases: list) -> bool:
        """Run test cases against code."""
        try:
            namespace = {}
            exec(code, namespace)
            
            # Find the defined function
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    func = obj
                    break
            
            if not func:
                return False
            
            # Run test cases
            for inputs, expected in test_cases:
                if isinstance(inputs, tuple):
                    result = func(*inputs)
                else:
                    result = func(inputs)
                
                if result != expected:
                    return False
            
            return True
        except:
            return False


class MathEvaluator(TaskEvaluator):
    """Evaluator for math problems."""
    
    def evaluate(self, query: str, response: str, expected: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate math answer.
        
        Args:
            query: Original query
            response: Model response
            expected: Expected answer
            
        Returns:
            Dictionary with 'exact_match', 'extracted_answer'
        """
        # Extract numerical answer from response
        answer = self._extract_answer(response)
        
        result = {
            'extracted_answer': answer,
            'expected_answer': expected,
        }
        
        if expected:
            # Normalize and compare
            result['exact_match'] = self._compare_answers(answer, expected)
        else:
            result['exact_match'] = None
        
        return result
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from response."""
        # Look for patterns like "The answer is X" or "= X"
        patterns = [
            r'(?:answer is|result is|equals?)\s*[:=]?\s*([-+]?\d+\.?\d*)',
            r'=\s*([-+]?\d+\.?\d*)\s*$',
            r'\b([-+]?\d+\.?\d*)\s*$',  # Last number in response
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, try to find any number
        numbers = re.findall(r'[-+]?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]  # Return last number found
        
        return None
    
    def _compare_answers(self, answer: Optional[str], expected: str) -> bool:
        """Compare numerical answers with tolerance."""
        if answer is None:
            return False
        
        try:
            ans_float = float(answer)
            exp_float = float(expected)
            
            # Allow small floating point differences
            return abs(ans_float - exp_float) < 0.01
        except:
            # Fall back to string comparison
            return answer.strip().lower() == expected.strip().lower()


class ReasoningEvaluator(TaskEvaluator):
    """Evaluator for reasoning/QA tasks."""
    
    def evaluate(self, query: str, response: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate reasoning quality.
        
        Uses ROUGE/BLEU if reference available, otherwise heuristics.
        
        Args:
            query: Original query
            response: Model response
            reference: Reference answer
            
        Returns:
            Quality metrics
        """
        result = {
            'length': len(response.split()),
            'has_response': len(response.strip()) > 0
        }
        
        if reference:
            # Simple word overlap (can be enhanced with ROUGE later)
            response_words = set(response.lower().split())
            reference_words = set(reference.lower().split())
            
            if len(reference_words) > 0:
                overlap = len(response_words & reference_words) / len(reference_words)
                result['word_overlap'] = overlap
        
        return result
