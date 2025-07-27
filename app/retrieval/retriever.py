"""
Retrieval module for finding relevant documents using hybrid search
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from retrieval process"""
    content: str
    score: float
    source_id: str
    metadata: Dict[str, Any]

class HybridRetriever:
    """
    Hybrid retriever combining dense vector search with optional reranking
    """
    
    def __init__(self, embedding_service, vector_store, similarity_threshold: float = 0.6):
        """
        Initialize the retriever
        
        Args:
            embedding_service: Embedding service for encoding queries
            vector_store: Vector store for similarity search
            similarity_threshold: Minimum similarity score for results
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
    
    def retrieve(self, query: str, top_k: int = 5, 
                filter_conditions: Optional[Dict] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of top results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            start_time = time.time()
            
            # Encode query to embedding
            query_embedding = self.embedding_service.encode_single(query)
            
            if query_embedding is None or len(query_embedding) == 0:
                logger.warning("Failed to encode query")
                return []
            
            # Search vector store
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more results for filtering
                filter_conditions=filter_conditions
            )
            
            # Filter by similarity threshold and convert to RetrievalResult
            results = []
            for result in search_results:
                if result.score >= self.similarity_threshold:
                    results.append(RetrievalResult(
                        content=result.content,
                        score=result.score,
                        source_id=result.id,
                        metadata=result.metadata
                    ))
            
            # Take only top_k results
            results = results[:top_k]
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []
    
    def retrieve_with_context(self, query: str, conversation_history: List[Dict] = None,
                            top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve documents with conversation context
        
        Args:
            query: Current user query
            conversation_history: Previous conversation turns
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            # Enhance query with conversation context
            enhanced_query = self._enhance_query_with_context(query, conversation_history)
            
            # Perform regular retrieval
            return self.retrieve(enhanced_query, top_k)
            
        except Exception as e:
            logger.error(f"Error during contextual retrieval: {str(e)}")
            # Fallback to regular retrieval
            return self.retrieve(query, top_k)
    
    def _enhance_query_with_context(self, query: str, 
                                  conversation_history: List[Dict] = None) -> str:
        """
        Enhance query with conversation context
        
        Args:
            query: Current query
            conversation_history: Previous conversation turns
            
        Returns:
            Enhanced query string
        """
        if not conversation_history:
            return query
        
        # Take last few turns for context
        recent_turns = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        
        context_parts = []
        for turn in recent_turns:
            if "user_query" in turn:
                context_parts.append(turn["user_query"])
        
        if context_parts:
            # Combine recent queries with current query
            enhanced_query = " ".join(context_parts + [query])
            return enhanced_query
        
        return query
    
    def rerank_results(self, query: str, results: List[RetrievalResult], 
                      rerank_top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Rerank results based on query-document similarity
        
        Args:
            query: Original query
            results: Initial retrieval results
            rerank_top_k: Number of top results to rerank (None for all)
            
        Returns:
            Reranked list of RetrievalResult objects
        """
        try:
            if not results:
                return results
            
            # If no reranking needed, return as is
            if rerank_top_k is None:
                return sorted(results, key=lambda x: x.score, reverse=True)
            
            # Take top results for reranking
            to_rerank = results[:rerank_top_k] if len(results) > rerank_top_k else results
            
            # Encode query and documents
            query_embedding = self.embedding_service.encode_single(query)
            document_texts = [result.content for result in to_rerank]
            document_embeddings = self.embedding_service.encode_batch(document_texts)
            
            # Calculate new similarities
            reranked_results = []
            for i, result in enumerate(to_rerank):
                if i < len(document_embeddings):
                    new_score = self.embedding_service.calculate_similarity(
                        query_embedding, document_embeddings[i]
                    )
                    result.score = max(result.score, new_score)  # Take the better score
                
                reranked_results.append(result)
            
            # Sort by new scores
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Reranked {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return results
    
    def get_retrieval_stats(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Get statistics about retrieval results
        
        Args:
            results: List of retrieval results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {"total_results": 0}
        
        scores = [result.score for result in results]
        content_lengths = [len(result.content) for result in results]
        
        return {
            "total_results": len(results),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_content_length": sum(content_lengths) / len(content_lengths),
            "above_threshold": len([s for s in scores if s >= self.similarity_threshold])
        }

# Global retriever instance
_retriever_instance = None

def get_retriever(embedding_service, vector_store, 
                 similarity_threshold: float = 0.6) -> HybridRetriever:
    """
    Get or create the global retriever instance
    
    Args:
        embedding_service: Embedding service
        vector_store: Vector store
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        HybridRetriever instance
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever(embedding_service, vector_store, similarity_threshold)
    
    return _retriever_instance
