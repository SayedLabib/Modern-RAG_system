from pydantic import BaseModel, Field
from typing import List, Optional

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    response: str = Field(..., description="Generated response")
    sources: List[str] = Field(default=[], description="Source chunks used for generation")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score of the response", ge=0.0, le=1.0)
    retrieval_time: Optional[float] = Field(default=None, description="Time taken for retrieval in seconds")
    generation_time: Optional[float] = Field(default=None, description="Time taken for generation in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "বাংলা সাহিত্যের ইতিহাস অত্যন্ত সমৃদ্ধ...",
                "sources": ["chunk_1", "chunk_2"],
                "confidence_score": 0.85,
                "retrieval_time": 0.1,
                "generation_time": 0.5
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Health status")
    embedding_service: bool = Field(..., description="Embedding service health")
    vector_store: bool = Field(..., description="Vector store health")
    llm_service: bool = Field(..., description="LLM service health")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "embedding_service": True,
                "vector_store": True,
                "llm_service": True
            }
        }
