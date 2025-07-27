"""
Main FastAPI application for the Multilingual RAG System
"""
import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from core.config import settings
from api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Multilingual RAG System...")
    
    try:
        # Import RAG service to trigger initialization
        from services.rag_service import get_rag_service
        rag_service = get_rag_service()
        
        if rag_service.is_initialized:
            logger.info("RAG system started successfully!")
        else:
            logger.warning("RAG system initialization incomplete")
        
    except Exception as e:
        logger.error(f"Failed to start RAG system: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multilingual RAG System...")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.description,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Multilingual RAG System",
        "version": settings.app_version,
        "docs": "/docs"
    }

if __name__ == "__main__":
    # Run the application using uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )