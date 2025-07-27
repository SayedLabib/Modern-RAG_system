import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Application Info
    app_name: str = "Multilingual RAG System"
    app_version: str = "1.0.0"
    description: str = "Multilingual RAG system for Bangla and English text"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "multilingual_knowledge_base"
    qdrant_vector_size: int = 384  # paraphrase-multilingual-MiniLM-L12-v2
    
    # Embedding Model Configuration
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"  # or "cuda" if GPU available
    
    # LLM Configuration (Groq)
    groq_api_key: Optional[str] = None
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # Default model
    groq_temperature: float = 0.7
    groq_max_tokens: int = 2048
    
    # Retrieval Configuration
    default_top_k: int = 5
    similarity_threshold: float = 0.3  # Lowered from 0.6 to allow more results
    max_chunk_length: int = 800
    chunk_overlap: int = 100
    
    # Data Paths
    data_dir: str = "data"  # Relative to app directory
    processed_chunks_file: str = "processed_chunks.txt"
    bangla_book_file: str = "Bangla_cleaned_book_improved.txt"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
