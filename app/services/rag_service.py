"""
Main RAG service that orchestrates the entire pipeline
"""
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import os
import sys

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.config import settings
from embeddings.embeddings import get_embedding_service
from vectorstore.qdrant_client import get_vector_store
from retrieval.retriever import get_retriever
from llm.generator import get_generator
from chunking.chunker import get_chunker

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from the RAG system"""
    response: str
    sources: List[str]
    confidence_score: float
    retrieval_time: float
    generation_time: float
    total_time: float

class RAGService:
    """
    Main RAG service that orchestrates document retrieval and response generation
    """
    
    def __init__(self):
        """Initialize the RAG service with all components"""
        self.embedding_service = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self.chunker = None
        self.is_initialized = False
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all required services"""
        try:
            logger.info("Initializing RAG services...")
            
            # Initialize embedding service
            self.embedding_service = get_embedding_service(
                model_name=settings.embedding_model_name,
                device=settings.embedding_device
            )
            
            # Initialize vector store
            self.vector_store = get_vector_store(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection_name=settings.qdrant_collection_name,
                vector_size=settings.qdrant_vector_size
            )
            
            # Initialize retriever
            self.retriever = get_retriever(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store,
                similarity_threshold=settings.similarity_threshold
            )
            
            # Initialize generator (if API key is available)
            if settings.groq_api_key:
                self.generator = get_generator(
                    api_key=settings.groq_api_key,
                    model=settings.groq_model,
                    temperature=settings.groq_temperature,
                    max_tokens=settings.groq_max_tokens
                )
            else:
                logger.warning("No Groq API key found. Generator will not be available.")
                self.generator = None
            
            # Initialize chunker
            self.chunker = get_chunker(
                chunk_size=settings.max_chunk_length,
                chunk_overlap=settings.chunk_overlap
            )
            
            self.is_initialized = True
            logger.info("RAG services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {str(e)}")
            self.is_initialized = False
            raise
    
    def process_query(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Process a user query and return a response
        
        Args:
            query: User query in Bangla or English
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse object
        """
        start_time = time.time()
        
        if not self.is_initialized:
            return RAGResponse(
                response="Error: RAG service not properly initialized",
                sources=[],
                confidence_score=0.0,
                retrieval_time=0.0,
                generation_time=0.0,
                total_time=0.0
            )
        
        try:
            if not query or not query.strip():
                return RAGResponse(
                    response="দয়া করে একটি প্রশ্ন লিখুন।",
                    sources=[],
                    confidence_score=0.0,
                    retrieval_time=0.0,
                    generation_time=0.0,
                    total_time=0.0
                )
            
            # Use provided top_k or default from settings
            k = top_k or settings.default_top_k
            
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(query, top_k=k)
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_docs:
                return RAGResponse(
                    response="দুঃখিত, আপনার প্রশ্নের জন্য কোনো প্রাসঙ্গিক তথ্য খুঁজে পাওয়া যায়নি।",
                    sources=[],
                    confidence_score=0.0,
                    retrieval_time=retrieval_time,
                    generation_time=0.0,
                    total_time=time.time() - start_time
                )
            
            # Step 2: Generate response using LLM
            generation_start = time.time()
            
            if self.generator:
                context_documents = [doc.content for doc in retrieved_docs]
                generation_result = self.generator.generate_response(
                    query=query,
                    context_documents=context_documents
                )
                generated_response = generation_result.response
                generation_time = generation_result.generation_time
            else:
                # Fallback: Simple concatenation of retrieved documents
                context_texts = [doc.content for doc in retrieved_docs]
                generated_response = f"প্রাসঙ্গিক তথ্য:\n\n" + "\n\n".join(context_texts)
                generation_time = time.time() - generation_start
            
            # Calculate confidence score (average of retrieval scores)
            confidence_score = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs)
            
            # Prepare sources
            sources = [f"Document {doc.source_id}: {doc.content[:100]}..." for doc in retrieved_docs]
            
            total_time = time.time() - start_time
            
            logger.info(f"Processed query in {total_time:.3f}s (retrieval: {retrieval_time:.3f}s, generation: {generation_time:.3f}s)")
            
            return RAGResponse(
                response=generated_response,
                sources=sources,
                confidence_score=confidence_score,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            total_time = time.time() - start_time
            
            return RAGResponse(
                response=f"দুঃখিত, আপনার প্রশ্ন প্রক্রিয়া করতে একটি ত্রুটি হয়েছে: {str(e)}",
                sources=[],
                confidence_score=0.0,
                retrieval_time=0.0,
                generation_time=0.0,
                total_time=total_time
            )
    
    def add_documents(self, file_path: str) -> bool:
        """
        Add documents from a file to the vector store
        
        Args:
            file_path: Path to the text file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                logger.error("RAG service not initialized")
                return False
            
            logger.info(f"Processing file: {file_path}")
            
            # Step 1: Chunk the document
            chunks = self.chunker.chunk_file(file_path)
            
            if not chunks:
                logger.error("No chunks created from file")
                return False
            
            # Step 2: Generate embeddings
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedding_service.encode_batch(chunk_texts)
            
            if len(embeddings) == 0:
                logger.error("Failed to generate embeddings")
                return False
            
            # Step 3: Add to vector store
            success = self.vector_store.add_documents(chunks, embeddings)
            
            if success:
                logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            else:
                logger.error("Failed to add chunks to vector store")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check the health of all services
        
        Returns:
            Dictionary with health status of each service
        """
        health_status = {
            "rag_service": self.is_initialized,
            "embedding_service": False,
            "vector_store": False,
            "generator": False
        }
        
        try:
            if self.embedding_service:
                health_status["embedding_service"] = self.embedding_service.health_check()
            
            if self.vector_store:
                health_status["vector_store"] = self.vector_store.health_check()
            
            if self.generator:
                health_status["generator"] = self.generator.health_check()
                
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}")
        
        return health_status
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "is_initialized": self.is_initialized,
            "embedding_model": settings.embedding_model_name,
            "llm_model": settings.groq_model,
            "vector_store_collection": settings.qdrant_collection_name
        }
        
        try:
            if self.embedding_service:
                stats["embedding_info"] = self.embedding_service.get_model_info()
            
            if self.vector_store:
                stats["collection_info"] = self.vector_store.get_collection_info()
                
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
        
        return stats

# Global RAG service instance
_rag_service_instance = None

def get_rag_service() -> RAGService:
    """
    Get or create the global RAG service instance
    
    Returns:
        RAGService instance
    """
    global _rag_service_instance
    
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    
    return _rag_service_instance
