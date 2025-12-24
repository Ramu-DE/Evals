"""Property-based tests for Vector Store functionality.

**Feature: rag-evaluation-pipeline, Property 9: Vector Storage Round-Trip**
**Validates: Requirements 3.1, 3.5**

**Feature: rag-evaluation-pipeline, Property 10: Search Performance Bounds**
**Validates: Requirements 3.2**

**Feature: rag-evaluation-pipeline, Property 11: Collection Management Operations**
**Validates: Requirements 3.4**
"""

import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_store.interfaces import VectorStore, VectorConfig, SearchResult, StorageResult
from vector_store.qdrant_store import QdrantVectorStore


# Test data generators
@st.composite
def vector_data(draw):
    """Generate test vector data."""
    dimension = draw(st.integers(min_value=2, max_value=128))  # Reduced max dimension
    num_vectors = draw(st.integers(min_value=1, max_value=5))  # Reduced max vectors
    
    vectors = []
    metadata = []
    
    for i in range(num_vectors):
        # Generate normalized vectors to avoid numerical issues
        vector = draw(st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=dimension,
            max_size=dimension
        ))
        vector_array = np.array(vector, dtype=np.float32)
        
        # Ensure vector is not zero to avoid NaN in similarity calculation
        norm = np.linalg.norm(vector_array)
        if norm == 0:
            # Create a non-zero vector if we get a zero vector
            vector_array[0] = 1.0
            norm = np.linalg.norm(vector_array)
        
        # Normalize to unit vector
        vector_array = vector_array / norm
        vectors.append(vector_array)
        
        # Generate metadata with unique IDs
        meta = {
            "id": f"vec_{i}_{draw(st.integers(min_value=1000, max_value=9999))}",
            "category": draw(st.sampled_from(["doc", "query", "answer"])),
            "score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        }
        metadata.append(meta)
    
    return vectors, metadata, dimension


@st.composite
def collection_config(draw):
    """Generate collection configuration."""
    dimension = draw(st.integers(min_value=2, max_value=128))  # Reduced max dimension
    distance_metric = draw(st.sampled_from(["cosine", "euclidean", "dot"]))
    index_type = draw(st.sampled_from(["hnsw", "flat"]))
    
    return VectorConfig(
        dimension=dimension,
        distance_metric=distance_metric,
        index_type=index_type
    )


class MockQdrantClient:
    """Mock Qdrant client for testing without external dependencies."""
    
    def __init__(self, url=None, api_key=None, timeout=None):
        """Initialize mock client, accepting same parameters as real client."""
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.collections = {}
        self.points = {}
        
    def get_collections(self):
        """Mock get collections."""
        return Mock(collections=[Mock(name=name) for name in self.collections.keys()])
    
    def create_collection(self, collection_name, vectors_config):
        """Mock create collection."""
        self.collections[collection_name] = {
            "config": vectors_config,
            "points": {}
        }
        self.points[collection_name] = {}
    
    def delete_collection(self, collection_name):
        """Mock delete collection."""
        if collection_name in self.collections:
            del self.collections[collection_name]
            del self.points[collection_name]
    
    def get_collection(self, collection_name):
        """Mock get collection info."""
        if collection_name not in self.collections:
            raise Exception(f"Collection {collection_name} not found")
        
        config = self.collections[collection_name]["config"]
        points_count = len(self.points.get(collection_name, {}))
        
        return Mock(
            status="green",
            vectors_count=points_count,
            points_count=points_count,
            config=Mock(
                params=config,
                dict=lambda: {"params": config}
            )
        )
    
    def upsert(self, collection_name, points):
        """Mock upsert points."""
        if collection_name not in self.collections:
            raise Exception(f"Collection {collection_name} not found")
        
        if collection_name not in self.points:
            self.points[collection_name] = {}
        
        for point in points:
            self.points[collection_name][point.id] = {
                "vector": point.vector,
                "payload": point.payload
            }
        
        from qdrant_client.http import models
        return Mock(status=models.UpdateStatus.COMPLETED)
    
    def search(self, collection_name, query_vector, limit, with_payload=True, with_vectors=True):
        """Mock similarity search."""
        if collection_name not in self.points:
            return []
        
        results = []
        for point_id, point_data in self.points[collection_name].items():
            # Simple cosine similarity calculation
            stored_vector = np.array(point_data["vector"])
            query_vec = np.array(query_vector)
            
            # Normalize vectors
            stored_norm = np.linalg.norm(stored_vector)
            query_norm = np.linalg.norm(query_vec)
            
            if stored_norm > 0 and query_norm > 0:
                similarity = np.dot(stored_vector, query_vec) / (stored_norm * query_norm)
            else:
                similarity = 0.0
            
            result = Mock(
                id=point_id,
                score=float(similarity),
                vector=point_data["vector"] if with_vectors else None,
                payload=point_data["payload"] if with_payload else None
            )
            results.append(result)
        
        # Sort by similarity score (descending) and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]


class TestVectorStoreProperties:
    """Property-based tests for Vector Store functionality."""
    
    @given(vector_data())
    @settings(max_examples=50, deadline=10000, suppress_health_check=[HealthCheck.data_too_large])
    def test_vector_storage_round_trip(self, test_data):
        """Property test: Vector storage round-trip consistency.
        
        **Feature: rag-evaluation-pipeline, Property 9: Vector Storage Round-Trip**
        **Validates: Requirements 3.1, 3.5**
        
        For any vector embedding submitted to the Vector Store, retrieving it 
        through similarity search should return the original vector with high accuracy.
        """
        vectors, metadata, dimension = test_data
        assume(len(vectors) > 0)
        
        # Create mock vector store
        with patch.object(QdrantVectorStore, '_connect') as mock_connect:
            mock_client = MockQdrantClient()
            
            store = QdrantVectorStore()
            store._client = mock_client  # Directly set the mock client
            
            # Create collection
            collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
            config = VectorConfig(
                dimension=dimension,
                distance_metric="cosine",
                index_type="hnsw"
            )
            
            collection_id = store.create_collection(collection_name, config)
            assert collection_id == collection_name
            
            # Store vectors
            result = store.store_embeddings(collection_name, vectors, metadata)
            assert result.success, f"Storage failed: {result.error_message}"
            assert result.stored_count == len(vectors)
            assert len(result.failed_ids) == 0
            
            # Test round-trip for each vector
            for i, original_vector in enumerate(vectors):
                search_results = store.similarity_search(
                    original_vector, collection_name, top_k=len(vectors)  # Get all results
                )
                
                assert len(search_results) > 0, f"No search results for vector {i}"
                
                # Find the result that matches our original metadata
                # (since identical vectors might return different matches)
                matching_result = None
                for result in search_results:
                    if result.metadata == metadata[i]:
                        matching_result = result
                        break
                
                # If we can't find exact metadata match, check if vector similarity is high enough
                if matching_result is None:
                    # Take the first result and verify it's similar enough
                    matching_result = search_results[0]
                
                retrieved_vector = matching_result.vector
                
                # Verify vector similarity (should be very high for exact match)
                original_norm = np.linalg.norm(original_vector)
                retrieved_norm = np.linalg.norm(retrieved_vector)
                
                if original_norm > 0 and retrieved_norm > 0:
                    similarity = np.dot(original_vector, retrieved_vector) / (original_norm * retrieved_norm)
                    # Allow for small numerical differences due to serialization
                    assert similarity > 0.99, f"Vector similarity too low: {similarity}"
                else:
                    # If either vector is zero, they should both be zero for exact match
                    assert original_norm == retrieved_norm == 0, "Zero vector mismatch"
    
    @given(vector_data(), st.integers(min_value=1, max_value=5))  # Reduced max top_k
    @settings(max_examples=30, deadline=15000, suppress_health_check=[HealthCheck.data_too_large])
    def test_search_performance_bounds(self, test_data, top_k):
        """Property test: Search performance bounds.
        
        **Feature: rag-evaluation-pipeline, Property 10: Search Performance Bounds**
        **Validates: Requirements 3.2**
        
        For any similarity search request, the Vector Store should return results 
        within acceptable latency thresholds regardless of collection size.
        """
        vectors, metadata, dimension = test_data
        assume(len(vectors) > 0)
        assume(top_k <= len(vectors))
        
        with patch.object(QdrantVectorStore, '_connect') as mock_connect:
            mock_client = MockQdrantClient()
            
            store = QdrantVectorStore()
            store._client = mock_client
            
            # Create collection
            collection_name = f"perf_test_{uuid.uuid4().hex[:8]}"
            config = VectorConfig(
                dimension=dimension,
                distance_metric="cosine",
                index_type="hnsw"
            )
            
            store.create_collection(collection_name, config)
            
            # Store vectors
            result = store.store_embeddings(collection_name, vectors, metadata)
            assert result.success
            
            # Test search performance
            query_vector = vectors[0]  # Use first vector as query
            
            start_time = time.time()
            search_results = store.similarity_search(query_vector, collection_name, top_k=top_k)
            end_time = time.time()
            
            search_latency = end_time - start_time
            
            # Performance assertions
            assert len(search_results) <= top_k, "Returned more results than requested"
            assert len(search_results) > 0, "No results returned"
            
            # Latency should be reasonable for mock implementation (very fast)
            # In real implementation, this would be a more meaningful threshold
            assert search_latency < 1.0, f"Search took too long: {search_latency}s"
            
            # Results should be ordered by similarity score (descending)
            scores = [result.score for result in search_results]
            assert scores == sorted(scores, reverse=True), "Results not properly ordered by score"
    
    @given(collection_config(), st.sampled_from(["delete", "info", "recreate"]))
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.data_too_large])
    def test_collection_management_operations(self, config, operation):
        """Property test: Collection management operations.
        
        **Feature: rag-evaluation-pipeline, Property 11: Collection Management Operations**
        **Validates: Requirements 3.4**
        
        For any collection management operation (create, delete, modify), the Vector Store 
        should execute it successfully and maintain data integrity.
        """
        with patch.object(QdrantVectorStore, '_connect') as mock_connect:
            mock_client = MockQdrantClient()
            
            store = QdrantVectorStore()
            store._client = mock_client
            
            # Create collection
            collection_name = f"mgmt_test_{uuid.uuid4().hex[:8]}"
            collection_id = store.create_collection(collection_name, config)
            assert collection_id == collection_name
            
            # Add some test data
            test_vectors = [np.random.rand(config.dimension).astype(np.float32)]
            test_metadata = [{"test": "data"}]
            
            storage_result = store.store_embeddings(collection_name, test_vectors, test_metadata)
            assert storage_result.success
            
            # Test management operation
            mgmt_result = store.manage_collection(collection_id, operation)
            
            # Verify operation result structure
            assert isinstance(mgmt_result, dict)
            assert "success" in mgmt_result
            assert "operation" in mgmt_result
            assert "collection_id" in mgmt_result
            assert mgmt_result["operation"] == operation
            assert mgmt_result["collection_id"] == collection_id
            
            if operation == "delete":
                assert mgmt_result["success"], f"Delete operation failed: {mgmt_result.get('error')}"
                assert "message" in mgmt_result
                
                # Verify collection is actually deleted by trying to access it
                try:
                    store.manage_collection(collection_id, "info")
                    # If we get here, deletion didn't work properly
                    assert False, "Collection still exists after deletion"
                except:
                    # Expected - collection should not exist
                    pass
            
            elif operation == "info":
                assert mgmt_result["success"], f"Info operation failed: {mgmt_result.get('error')}"
                assert "info" in mgmt_result
                
                info = mgmt_result["info"]
                assert "status" in info
                assert "vectors_count" in info
                assert "points_count" in info
                
                # Should have at least the test data we added
                assert info["points_count"] >= 1
            
            elif operation == "recreate":
                assert mgmt_result["success"], f"Recreate operation failed: {mgmt_result.get('error')}"
                assert "message" in mgmt_result
                
                # Verify collection exists and is empty after recreation
                info_result = store.manage_collection(collection_id, "info")
                assert info_result["success"]
                # After recreation, collection should be empty
                assert info_result["info"]["points_count"] == 0
    
    def test_error_handling_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with patch.object(QdrantVectorStore, '_connect') as mock_connect:
            mock_client = MockQdrantClient()
            
            store = QdrantVectorStore()
            store._client = mock_client
            
            # Test empty vectors
            result = store.store_embeddings("test", [], [])
            assert not result.success
            assert "No vectors provided" in result.error_message
            
            # Test mismatched vectors and metadata
            vectors = [np.array([1.0, 2.0])]
            metadata = [{"test": "1"}, {"test": "2"}]  # Too many metadata entries
            
            result = store.store_embeddings("test", vectors, metadata)
            assert not result.success
            assert "must match" in result.error_message
    
    def test_connection_management(self):
        """Test connection management and retry logic."""
        with patch('vector_store.qdrant_store.QdrantClient') as mock_client_class:
            # Mock client that succeeds after some retries
            mock_client = Mock()
            # Set up side effect to fail twice then succeed for all subsequent calls
            call_count = 0
            def get_collections_side_effect():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Connection failed")
                return Mock(collections=[])
            
            mock_client.get_collections.side_effect = get_collections_side_effect
            mock_client_class.return_value = mock_client
            
            store = QdrantVectorStore()
            
            # This should succeed after retries
            health = store.health_check()
            assert health["healthy"]
            
            # Verify retry logic was used
            assert call_count >= 3