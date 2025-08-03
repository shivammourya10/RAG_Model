"""
Simple embedder that uses numpy for basic embedding functionality 
when SentenceTransformer has meta tensor issues.
"""

import numpy as np
from typing import List, Union

class SimpleEmbedder:
    """Ultra simple embedder that uses hash-based embeddings."""
    
    def __init__(self, dimension=384):
        """Initialize embedder with specified dimension."""
        self.dimension = dimension
        print("🔄 SimpleEmbedder initialized - Using hash-based embeddings")
    
    def encode(self, texts: Union[str, List[str]], **kwargs):
        """Generate embeddings for text inputs."""
        if isinstance(texts, str):
            return self._encode_single(texts)
        else:
            return np.array([self._encode_single(text) for text in texts])
    
    def _encode_single(self, text: str):
        """Generate a deterministic embedding for a single text."""
        # Generate a seed from the text hash
        text_hash = hash(text)
        np.random.seed(text_hash)
        
        # Generate a normalized embedding vector
        embedding = np.random.normal(0, 1, self.dimension)
        return embedding / np.linalg.norm(embedding)

def get_simple_embedder(dimension=384):
    """Get a simple embedder instance."""
    return SimpleEmbedder(dimension=dimension)
