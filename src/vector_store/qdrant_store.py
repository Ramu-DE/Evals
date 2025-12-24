"""Qdrant implementation of the Vector Store interface."""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
import uuid
import time

from .interfaces import VectorStore, VectorConfig, SearchResult, StorageResult
from config import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of the Vector Store interface."""
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize Qdrant client with connection management.
        
        Args:
            url: Qdrant cluster URL. If None, uses settings.qdrant_url
            api_key: API key for authentication. If None, uses settings.qdrant_api_key
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self._client = None
        self._connection_retries = 3
        self._retry_delay = 1.0
        
    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client with connection management."""
        if self._client is None:
            self._connect()
        return self._client
    
    def _connect(self) -> None:
        """Establish connection to Qdrant cluster with retry logic."""
        for attempt in range(self._connection_retries):
            try:
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=30.0
                )
                # Test connection
                self._client.get_collections()
                logger.info(f"Successfully connected to Qdrant at {self.url}")
                return
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self._connection_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise ConnectionError(f"Failed to connect to Qdrant after {self._connection_retries} attempts: {e}")
    
    def _reconnect_if_needed(self) -> None:
        """Reconnect if the current connection is stale."""
        try:
            if self._client:
                self._client.get_collections()
        except Exception:
            logger.info("Connection appears stale, reconnecting...")
            self._client = None
            self._connect()
    
    def store_embeddings(self, collection_name: str, vectors: List[np.ndarray], 
                        metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> StorageResult:
        """Store vector embeddings with metadata in Qdrant.
        
        Args:
            collection_name: Name of the collection to store vectors in
            vectors: List of numpy arrays representing vectors
            metadata: List of metadata dictionaries for each vector
            ids: Optional list of IDs. If None, UUIDs will be generated
            
        Returns:
            StorageResult with success status and details
        """
        if not vectors:
            return StorageResult(
                success=False,
                stored_count=0,
                failed_ids=[],
                error_message="No vectors provided"
            )
        
        if len(vectors) != len(metadata):
            return StorageResult(
                success=False,
                stored_count=0,
                failed_ids=[],
                error_message="Number of vectors must match number of metadata entries"
            )
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        elif len(ids) != len(vectors):
            return StorageResult(
                success=False,
                stored_count=0,
                failed_ids=[],
                error_message="Number of IDs must match number of vectors"
            )
        
        try:
            self._reconnect_if_needed()
            
            # Prepare points for Qdrant
            points = []
            for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=meta
                ))
            
            # Upsert points to collection
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            if operation_info.status == models.UpdateStatus.COMPLETED:
                return StorageResult(
                    success=True,
                    stored_count=len(vectors),
                    failed_ids=[],
                    error_message=None
                )
            else:
                return StorageResult(
                    success=False,
                    stored_count=0,
                    failed_ids=ids,
                    error_message=f"Upsert operation failed with status: {operation_info.status}"
                )
                
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return StorageResult(
                success=False,
                stored_count=0,
                failed_ids=ids,
                error_message=str(e)
            )
    
    def similarity_search(self, query_vector: np.ndarray, collection_name: str, 
                         top_k: int = 10) -> List[SearchResult]:
        """Perform similarity search in Qdrant.
        
        Args:
            query_vector: Query vector as numpy array
            collection_name: Name of the collection to search in
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            self._reconnect_if_needed()
            
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
                with_vectors=True
            )
            
            results = []
            for point in search_result:
                results.append(SearchResult(
                    id=str(point.id),
                    score=point.score,
                    vector=np.array(point.vector),
                    metadata=point.payload or {}
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def create_collection(self, name: str, vector_config: VectorConfig) -> str:
        """Create a new vector collection in Qdrant.
        
        Args:
            name: Name of the collection
            vector_config: Configuration for the vector collection
            
        Returns:
            Collection ID (same as name for Qdrant)
        """
        try:
            self._reconnect_if_needed()
            
            # Map distance metrics
            distance_map = {
                "cosine": models.Distance.COSINE,
                "euclidean": models.Distance.EUCLID,
                "dot": models.Distance.DOT
            }
            
            distance = distance_map.get(vector_config.distance_metric.lower(), models.Distance.COSINE)
            
            # Create collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_config.dimension,
                    distance=distance
                )
            )
            
            logger.info(f"Successfully created collection '{name}' with dimension {vector_config.dimension}")
            return name
            
        except Exception as e:
            logger.error(f"Error creating collection '{name}': {e}")
            raise RuntimeError(f"Failed to create collection: {e}")
    
    def manage_collection(self, collection_id: str, operation: str) -> Dict[str, Any]:
        """Manage collection operations in Qdrant.
        
        Args:
            collection_id: ID/name of the collection
            operation: Operation to perform ('delete', 'info', 'recreate')
            
        Returns:
            Dictionary with operation result
        """
        try:
            self._reconnect_if_needed()
            
            if operation.lower() == "delete":
                self.client.delete_collection(collection_name=collection_id)
                return {
                    "success": True,
                    "operation": "delete",
                    "collection_id": collection_id,
                    "message": f"Collection '{collection_id}' deleted successfully"
                }
            
            elif operation.lower() == "info":
                collection_info = self.client.get_collection(collection_name=collection_id)
                return {
                    "success": True,
                    "operation": "info",
                    "collection_id": collection_id,
                    "info": {
                        "status": collection_info.status,
                        "vectors_count": collection_info.vectors_count,
                        "points_count": collection_info.points_count,
                        "config": collection_info.config.dict() if collection_info.config else None
                    }
                }
            
            elif operation.lower() == "recreate":
                # Get current collection info first
                try:
                    collection_info = self.client.get_collection(collection_name=collection_id)
                    vector_config = collection_info.config.params
                    
                    # Delete and recreate
                    self.client.delete_collection(collection_name=collection_id)
                    self.client.create_collection(
                        collection_name=collection_id,
                        vectors_config=vector_config
                    )
                    
                    return {
                        "success": True,
                        "operation": "recreate",
                        "collection_id": collection_id,
                        "message": f"Collection '{collection_id}' recreated successfully"
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "operation": "recreate",
                        "collection_id": collection_id,
                        "error": f"Failed to recreate collection: {e}"
                    }
            
            else:
                return {
                    "success": False,
                    "operation": operation,
                    "collection_id": collection_id,
                    "error": f"Unsupported operation: {operation}"
                }
                
        except Exception as e:
            logger.error(f"Error managing collection '{collection_id}' with operation '{operation}': {e}")
            return {
                "success": False,
                "operation": operation,
                "collection_id": collection_id,
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the Qdrant connection.
        
        Returns:
            Dictionary with health status information
        """
        try:
            self._reconnect_if_needed()
            collections = self.client.get_collections()
            
            return {
                "healthy": True,
                "url": self.url,
                "collections_count": len(collections.collections),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "healthy": False,
                "url": self.url,
                "error": str(e),
                "timestamp": time.time()
            }