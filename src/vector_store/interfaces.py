"""Base interfaces for the Vector Store component."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class VectorConfig:
    """Configuration for vector collections."""
    dimension: int
    distance_metric: str
    index_type: str


@dataclass
class SearchResult:
    """Result of a similarity search."""
    id: str
    score: float
    vector: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class StorageResult:
    """Result of a vector storage operation."""
    success: bool
    stored_count: int
    failed_ids: List[str]
    error_message: Optional[str]


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def store_embeddings(self, collection_name: str, vectors: List[np.ndarray], 
                        metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> StorageResult:
        """Store vector embeddings with metadata."""
        pass
    
    @abstractmethod
    def similarity_search(self, query_vector: np.ndarray, collection_name: str, 
                         top_k: int = 10) -> List[SearchResult]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    def create_collection(self, name: str, vector_config: VectorConfig) -> str:
        """Create a new vector collection."""
        pass
    
    @abstractmethod
    def manage_collection(self, collection_id: str, operation: str) -> Dict[str, Any]:
        """Manage collection operations (delete, modify, etc.)."""
        pass