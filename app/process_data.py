"""
Data processing script to load and index documents into the vector store
"""
import os
import sys
import logging
from pathlib import Path

# Add app directory to path
sys.path.append(os.path.dirname(__file__))

from services.rag_service import get_rag_service
from core.config import settings

logger = logging.getLogger(__name__)

def process_data_file(file_path: str) -> bool:
    """
    Process a data file and add it to the vector store
    
    Args:
        file_path: Path to the data file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize RAG service
        rag_service = get_rag_service()
        
        if not rag_service.is_initialized:
            logger.error("RAG service not initialized")
            return False
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Process the file
        logger.info(f"Processing file: {file_path}")
        success = rag_service.add_documents(file_path)
        
        if success:
            logger.info(f"Successfully processed {file_path}")
        else:
            logger.error(f"Failed to process {file_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return False

def main():
    """Main function to process all data files"""
    logging.basicConfig(level=logging.INFO)
    
    # Default data file path
    data_file = os.path.join(settings.data_dir, settings.bangla_book_file)
    
    # Check if data file exists
    if os.path.exists(data_file):
        logger.info(f"Found data file: {data_file}")
        success = process_data_file(data_file)
        
        if success:
            logger.info("Data processing completed successfully!")
        else:
            logger.error("Data processing failed!")
    else:
        logger.warning(f"Data file not found: {data_file}")
        logger.info("Please place your data file in the app/data directory")
        
        # List available files in data directory
        data_dir = Path(settings.data_dir)
        if data_dir.exists():
            files = list(data_dir.glob("*.txt"))
            if files:
                logger.info("Available text files:")
                for file in files:
                    logger.info(f"  - {file}")
                    
                # Process the first available file
                first_file = str(files[0])
                logger.info(f"Processing first available file: {first_file}")
                process_data_file(first_file)

if __name__ == "__main__":
    main()
