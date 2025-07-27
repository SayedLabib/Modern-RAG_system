from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="User query in Bangla or English", min_length=1)
    top_k: Optional[int] = Field(default=5, description="Number of top results to retrieve", ge=1, le=20)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলুন",
                "top_k": 5
            }
        }
