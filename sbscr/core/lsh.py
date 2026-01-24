"""
LSH (Locality-Sensitive Hashing) signature generator for semantic bucketing.
Uses MinHash for efficient similarity estimation.
"""

from typing import List, Set
import mmh3
from datasketch import MinHash
import re


class LSHSignatureGenerator:
    """Generate LSH signatures from text queries for fast semantic bucketing."""
    
    def __init__(self, num_perm: int = 128, ngram_size: int = 3, seed: int = 42):
        """
        Initialize LSH signature generator.
        
        Args:
            num_perm: Number of hash permutations (higher = more accurate, slower)
            ngram_size: Character n-gram size for hashing
            seed: Random seed for reproducibility
        """
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.seed = seed
        
    def _preprocess(self, text: str) -> str:
        """Normalize and clean text."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip
        text = text.strip()
        return text
    
    def _generate_ngrams(self, text: str) -> Set[str]:
        """Generate character n-grams from text."""
        text = self._preprocess(text)
        ngrams = set()
        
        # Character n-grams
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        
        # Also add word tokens for better semantic representation
        words = text.split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                ngrams.add(f"_WORD_{word}")
        
        return ngrams
    
    def generate_signature(self, query: str) -> MinHash:
        """
        Generate MinHash signature for a query.
        
        Args:
            query: Input text query
            
        Returns:
            MinHash signature object
        """
        minhash = MinHash(num_perm=self.num_perm, seed=self.seed)
        ngrams = self._generate_ngrams(query)
        
        for ngram in ngrams:
            minhash.update(ngram.encode('utf-8'))
        
        return minhash
    
    def estimate_similarity(self, sig1: MinHash, sig2: MinHash) -> float:
        """
        Estimate Jaccard similarity between two signatures.
        
        Args:
            sig1: First MinHash signature
            sig2: Second MinHash signature
            
        Returns:
            Similarity score between 0 and 1
        """
        return sig1.jaccard(sig2)
    
    def get_signature_hash(self, query: str) -> int:
        """
        Get a compact hash representation of the query signature.
        Useful for bucketing queries into discrete categories.
        
        Args:
            query: Input text query
            
        Returns:
            Integer hash value
        """
        minhash = self.generate_signature(query)
        # Convert first few hash values to a single integer
        hashvalues = minhash.hashvalues[:8]  # Use first 8 for bucketing
        combined = ''.join(str(h) for h in hashvalues)
        return mmh3.hash(combined, seed=self.seed)
    
    def get_bucket_id(self, query: str, num_buckets: int = 100) -> int:
        """
        Assign query to a semantic bucket.
        
        Args:
            query: Input text query
            num_buckets: Number of buckets to use
            
        Returns:
            Bucket ID (0 to num_buckets-1)
        """
        hash_val = self.get_signature_hash(query)
        return abs(hash_val) % num_buckets


class LSHIndex:
    """
    LSH index for fast nearest neighbor search.
    Maps queries to semantic buckets.
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.5):
        """
        Initialize LSH index.
        
        Args:
            num_perm: Number of hash permutations
            threshold: Similarity threshold for matching
        """
        from datasketch import MinHashLSH
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.generator = LSHSignatureGenerator(num_perm=num_perm)
        self.signatures = {}
        
    def insert(self, key: str, query: str):
        """
        Insert a query into the index.
        
        Args:
            key: Unique identifier for this query
            query: Query text
        """
        signature = self.generator.generate_signature(query)
        self.lsh.insert(key, signature)
        self.signatures[key] = signature
        
    def query(self, query: str) -> List[str]:
        """
        Find similar queries in the index.
        
        Args:
            query: Query text to search for
            
        Returns:
            List of matching keys
        """
        signature = self.generator.generate_signature(query)
        return self.lsh.query(signature)
