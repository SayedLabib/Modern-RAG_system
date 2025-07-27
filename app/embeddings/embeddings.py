import numpy as np
import numpy as np
from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class BanglaEmbeddingService:
    """
    Embedding service optimized for Bangla and multilingual text processing.
    Uses paraphrase-multilingual-MiniLM-L12-v2 for high-quality embeddings.
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dimension = 384  # Default for MiniLM-L12-v2
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get actual embedding dimension
            test_embedding = self.model.encode("test")
            self.embedding_dimension = len(test_embedding)
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding vector.
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for encoding")
            return np.zeros(self.embedding_dimension)
        
        try:
            # Preprocess text for better embedding quality
            processed_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(processed_text, convert_to_numpy=True)
            
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return np.zeros(self.embedding_dimension)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts in batches for efficiency.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
        
        try:
            # Preprocess all texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100
            )
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / np.maximum(norms, 1e-8)
            
            return normalized_embeddings
            
        except Exception as e:
            logger.error(f"Error encoding batch: {str(e)}")
            return np.zeros((len(texts), self.embedding_dimension))
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better embedding quality.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Limit text length to avoid memory issues
        max_length = 512  # Token limit for the model
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            # Calculate similarities with all candidates
            similarities = np.dot(candidate_embeddings, query_embedding)
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return (index, similarity) pairs
            results = [(int(idx), float(similarities[idx])) for idx in top_indices]
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar embeddings: {str(e)}")
            return []
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 512),
            "is_loaded": self.model is not None
        }
    
    def health_check(self) -> bool:
        """
        Check if the embedding service is working properly.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test encoding
            test_embedding = self.encode_single("স্বাস্থ্য পরীক্ষা")
            
            # Check if embedding is valid
            if len(test_embedding) != self.embedding_dimension:
                return False
            
            # Check if embedding contains valid numbers
            if np.isnan(test_embedding).any() or np.isinf(test_embedding).any():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

# Global embedding service instance
embedding_service = None

def get_embedding_service(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", 
                         device: str = "cpu") -> BanglaEmbeddingService:
    """
    Get or create the global embedding service instance.
    
    Args:
        model_name: Name of the embedding model
        device: Device to run the model on
        
    Returns:
        BanglaEmbeddingService instance
    """
    global embedding_service
    
    if embedding_service is None:
        embedding_service = BanglaEmbeddingService(model_name, device)
    
    return embedding_service
