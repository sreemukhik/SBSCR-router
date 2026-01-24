"""
Dataset loaders for evaluation benchmarks.
"""

from typing import List, Dict
import json


def load_humaneval_subset(n: int = 50) -> List[Dict]:
    """
    Load subset of HumanEval coding problems.
    
    Args:
        n: Number of problems to load (max 164)
        
    Returns:
        List of query dictionaries
    """
    # For now, create synthetic HumanEval-style queries
    # TODO: Replace with actual HumanEval dataset when available
    
    humaneval_examples = [
        {
            'query': 'Write a function to check if a number is prime',
            'domain': 'code',
            'complexity': 3.0,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Implement a function to find the longest common subsequence of two strings',
            'domain': 'code',
            'complexity': 6.0,
            'expected_model': 'deepseek-coder-v2'
        },
        {
            'query': 'Write a function that returns the first n Fibonacci numbers',
            'domain': 'code',
            'complexity': 2.5,
            'expected_model': 'phi-3-mini'
        },
        {
            'query': 'Implement a binary search algorithm for a sorted array',
            'domain': 'code',
            'complexity': 4.0,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Create a function to reverse a linked list in-place',
            'domain': 'code',
            'complexity': 5.0,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Write a function to merge two sorted lists',
            'domain': 'code',
            'complexity': 3.5,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Implement a depth-first search algorithm for a graph',
            'domain': 'code',
            'complexity': 6.5,
            'expected_model': 'deepseek-coder-v2'
        },
        {
            'query': 'Create a function that finds all permutations of a string',
            'domain': 'code',
            'complexity': 5.5,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Write a simple hello world function',
            'domain': 'code',
            'complexity': 1.0,
            'expected_model': 'phi-3-mini'
        },
        {
            'query': 'Implement a function to validate balanced parentheses',
            'domain': 'code',
            'complexity': 4.5,
            'expected_model': 'llama-3-8b'
        },
    ]
    
    return humaneval_examples[:min(n, len(humaneval_examples))]


def load_gsm8k_subset(n: int = 100) -> List[Dict]:
    """
    Load subset of GSM8K math problems.
    
    Args:
        n: Number of problems to load
        
    Returns:
        List of query dictionaries
    """
    gsm8k_examples = [
        {
            'query': 'What is 15 + 27?',
            'domain': 'math',
            'complexity': 1.0,
            'expected_model': 'phi-3-mini'
        },
        {
            'query': 'Calculate the area of a circle with radius 5',
            'domain': 'math',
            'complexity': 2.5,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Solve for x: 2x + 5 = 15',
            'domain': 'math',
            'complexity': 2.0,
            'expected_model': 'phi-3-mini'
        },
        {
            'query': 'Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1',
            'domain': 'math',
            'complexity': 4.5,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Calculate the sum of integers from 1 to 100',
            'domain': 'math',
            'complexity': 2.0,
            'expected_model': 'phi-3-mini'
        },
        {
            'query': 'If John has 5 apples and buys 3 more, how many does he have?',
            'domain': 'math',
            'complexity': 1.0,
            'expected_model': 'phi-3-mini'
        },
        {
            'query': 'Solve the system of equations: x + y = 10, x - y = 2',
            'domain': 'math',
            'complexity': 5.0,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Calculate the probability of rolling two dice and getting a sum of 7',
            'domain': 'math',
            'complexity': 4.0,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Find the integral of sin(x) from 0 to π',
            'domain': 'math',
            'complexity': 6.0,
            'expected_model': 'gemini-1.5-pro'
        },
        {
            'query': 'What is 25% of 80?',
            'domain': 'math',
            'complexity': 1.5,
            'expected_model': 'phi-3-mini'
        },
    ]
    
    return gsm8k_examples[:min(n, len(gsm8k_examples))]


def load_custom_dataset() -> List[Dict]:
    """
    Load custom mixed dataset with diverse query types.
    
    Returns:
        List of query dictionaries
    """
    custom_queries = [
        # Simple queries
        {
            'query': 'What is machine learning?',
            'domain': 'general',
            'complexity': 1.5,
            'expected_model': 'phi-3-mini'
        },
        {
            'query': 'Define recursion',
            'domain': 'general',
            'complexity': 1.0,
            'expected_model': 'phi-3-mini'
        },
        
        # Medium complexity
        {
            'query': 'Explain how neural networks work',
            'domain': 'reasoning',
            'complexity': 5.0,
            'expected_model': 'llama-3-8b'
        },
        {
            'query': 'Compare Python and JavaScript for web development',
            'domain': 'reasoning',
            'complexity': 4.5,
            'expected_model': 'llama-3-8b'
        },
        
        # Complex queries
        {
            'query': 'Design a scalable microservices architecture for an e-commerce platform',
            'domain': 'reasoning',
            'complexity': 8.5,
            'expected_model': 'gemini-1.5-pro'
        },
        {
            'query': 'Implement a distributed consensus algorithm using Raft protocol',
            'domain': 'code',
            'complexity': 9.0,
            'expected_model': 'gemini-1.5-pro'
        },
        {
            'query': 'Prove that the halting problem is undecidable',
            'domain': 'reasoning',
            'complexity': 9.5,
            'expected_model': 'gemini-1.5-pro'
        },
        
        # Creative
        {
            'query': 'Write a short story about a robot learning to feel emotions',
            'domain': 'creative',
            'complexity': 5.0,
            'expected_model': 'gemini-1.5-pro'
        },
        
        # Edge cases
        {
            'query': 'Hi',
            'domain': 'general',
            'complexity': 0.5,
            'expected_model': 'phi-3-mini'
        },
        {
            'query': 'Optimize a PostgreSQL query with 10 joins, 5 subqueries, and complex aggregations for a billion-row table',
            'domain': 'code',
            'complexity': 8.0,
            'expected_model': 'gemini-1.5-pro'
        },
    ]
    
    return custom_queries


def load_all_datasets() -> Dict[str, List[Dict]]:
    """
    Load all evaluation datasets.
    
    Returns:
        Dictionary mapping dataset names to query lists
    """
    return {
        'humaneval': load_humaneval_subset(10),
        'gsm8k': load_gsm8k_subset(10),
        'custom': load_custom_dataset()
    }
