"""
API routes for the RAG system
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import logging
import sys
import os

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.request import QueryRequest
from models.response import QueryResponse, HealthResponse
from services.rag_service import get_rag_service

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["RAG System"])

# Initialize RAG service
rag_service = get_rag_service()

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Process a user query and return a generated response
    
    Args:
        request: Query request containing the user's question
        
    Returns:
        QueryResponse with generated answer and metadata
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Process the query using RAG service
        rag_response = rag_service.process_query(
            query=request.query,
            top_k=request.top_k
        )
        
        # Convert to API response format
        response = QueryResponse(
            response=rag_response.response,
            sources=rag_response.sources,
            confidence_score=rag_response.confidence_score,
            retrieval_time=rag_response.retrieval_time,
            generation_time=rag_response.generation_time
        )
        
        logger.info("Query processed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health status of all RAG system components
    
    Returns:
        HealthResponse with status of each component
    """
    try:
        # Get health status from RAG service
        health_status = rag_service.health_check()
        
        # Determine overall status
        all_healthy = all(health_status.values())
        overall_status = "healthy" if all_healthy else "unhealthy"
        
        response = HealthResponse(
            status=overall_status,
            embedding_service=health_status.get("embedding_service", False),
            vector_store=health_status.get("vector_store", False),
            llm_service=health_status.get("generator", False)
        )
        
        logger.info(f"Health check completed - Status: {overall_status}")
        return response
        
    except Exception as e:
        logger.error(f"Error during health check: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/stats")
async def get_system_stats() -> Dict[str, Any]:
    """
    Get system statistics and information
    
    Returns:
        Dictionary with system statistics
    """
    try:
        stats = rag_service.get_stats()
        logger.info("System stats retrieved successfully")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system stats: {str(e)}"
        )
