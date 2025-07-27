"""
Qdrant vector store client for storing and retrieving embeddings
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result from vector store"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class QdrantVectorStore:
    """
    Qdrant vector store for storing and retrieving document embeddings
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_name: str = "multilingual_knowledge_base",
                 vector_size: int = 384):
        """
        Initialize Qdrant client
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection
            vector_size: Size of embedding vectors
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = None
        
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant and ensure collection exists"""
        try:
            # Connect to Qdrant
            self.client = QdrantClient(host=self.host, port=self.port, timeout=30.0)
            
            # Check if collection exists, create if not
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """
        Add documents with their embeddings to the vector store
        
        Args:
            documents: List of document dictionaries with content and metadata
            embeddings: Numpy array of embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Try with simple numeric IDs instead of UUIDs
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Use simple numeric ID
                point_id = i + 1
                
                payload = {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {})
                }
                
                # Ensure embedding is a list of floats
                vector = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                
                # Debug: Check first point structure
                if i == 0:
                    logger.info(f"Sample point structure - ID: {point_id}, vector length: {len(vector)}, payload keys: {list(payload.keys())}")
                
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            # Upload in smaller batches to avoid any issues
            batch_size = 10
            total_uploaded = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                try:
                    result = self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True
                    )
                    total_uploaded += len(batch)
                    logger.info(f"Uploaded batch {i//batch_size + 1}: {len(batch)} documents")
                except Exception as batch_error:
                    logger.error(f"Failed to upload batch {i//batch_size + 1}: {batch_error}")
                    return False
            
            logger.info(f"Successfully added {total_uploaded} documents to vector store")
            return total_uploaded > 0
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filter_conditions: Optional[Dict] = None) -> List[SearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Prepare filter if provided
            query_filter = None
            if filter_conditions:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        ) for key, value in filter_conditions.items()
                    ]
                )
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                results.append(SearchResult(
                    id=str(result.id),
                    content=result.payload.get("content", ""),
                    score=float(result.score),
                    metadata=result.payload.get("metadata", {})
                ))
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[SearchResult]:
        """
        Get a specific document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            SearchResult object or None
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True,
                with_vectors=False
            )
            
            if result:
                point = result[0]
                return SearchResult(
                    id=str(point.id),
                    content=point.payload.get("content", ""),
                    score=1.0,  # Perfect match for exact retrieval
                    metadata=point.payload.get("metadata", {})
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents by IDs
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=doc_ids
            )
            
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Collection information dictionary
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def health_check(self) -> bool:
        """
        Check if the vector store is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collection info
            self.client.get_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(self.collection_name)
            
            # Recreate the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Collection {self.collection_name} cleared and recreated")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

# Global vector store instance
_vector_store_instance = None

def get_vector_store(host: str = "localhost", port: int = 6333,
                    collection_name: str = "multilingual_knowledge_base",
                    vector_size: int = 384) -> QdrantVectorStore:
    """
    Get or create the global vector store instance
    
    Args:
        host: Qdrant server host
        port: Qdrant server port
        collection_name: Name of the collection
        vector_size: Size of embedding vectors
        
    Returns:
        QdrantVectorStore instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = QdrantVectorStore(host, port, collection_name, vector_size)
    
    return _vector_store_instance
