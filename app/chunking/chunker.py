"""
Text chunking module using LangChain's RecursiveCharacterTextSplitter
Optimized for Bangla and English mixed text
"""
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class BanglaTextChunker:
    """
    Text chunker optimized for Bangla and multilingual content using LangChain
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Custom separators for Bangla text
        separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            "।",     # Bangla sentence end
            ".",     # English sentence end
            "?",     # Question marks
            "!",     # Exclamation marks
            ";",     # Semicolons
            ",",     # Commas
            " ",     # Spaces
            "",      # Character level
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller segments
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            # Clean the text
            cleaned_text = self._preprocess_text(text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    chunk_objects.append({
                        "id": f"chunk_{i:04d}",
                        "content": chunk.strip(),
                        "length": len(chunk),
                        "index": i,
                        "metadata": {
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "total_chunks": len(chunks)
                        }
                    })
            
            logger.info(f"Created {len(chunk_objects)} chunks from text")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return []
    
    def chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunk text from a file
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of chunks with metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = self.chunk_text(text)
            
            # Add file metadata
            for chunk in chunks:
                chunk["metadata"]["source_file"] = os.path.basename(file_path)
                chunk["metadata"]["file_path"] = file_path
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better chunking
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Remove multiple consecutive newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        
        return text.strip()
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks
        
        Args:
            chunks: List of chunk objects
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {"total_chunks": 0}
        
        lengths = [chunk["length"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(lengths),
            "avg_chunk_length": sum(lengths) / len(lengths),
            "min_chunk_length": min(lengths),
            "max_chunk_length": max(lengths),
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap
        }

# Global chunker instance
_chunker_instance = None

def get_chunker(chunk_size: int = 800, chunk_overlap: int = 100) -> BanglaTextChunker:
    """
    Get or create the global chunker instance
    
    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        BanglaTextChunker instance
    """
    global _chunker_instance
    
    if _chunker_instance is None:
        _chunker_instance = BanglaTextChunker(chunk_size, chunk_overlap)
    
    return _chunker_instance
