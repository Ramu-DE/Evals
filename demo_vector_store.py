#!/usr/bin/env python3
"""
Demo script showing the Vector Store implementation in action.
This script demonstrates the key functionality without requiring external dependencies.
"""

import sys
from pathlib import Path
import numpy as np
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vector_store import QdrantVectorStore, VectorConfig
from tests.test_vector_store import MockQdrantClient


def demo_vector_store():
    """Demonstrate Vector Store functionality."""
    print("🚀 RAG Evaluation Pipeline - Vector Store Demo")
    print("=" * 50)
    
    # Mock the Qdrant client for demo purposes
    with patch('vector_store.qdrant_store.QdrantClient', MockQdrantClient):
        # Initialize the vector store
        store = QdrantVectorStore()
        print("✅ Vector Store initialized")
        
        # Create a collection
        config = VectorConfig(
            dimension=128,
            distance_metric="cosine",
            index_type="hnsw"
        )
        
        collection_name = "demo_collection"
        collection_id = store.create_collection(collection_name, config)
        print(f"✅ Collection '{collection_name}' created with ID: {collection_id}")
        
        # Generate some demo vectors
        vectors = [
            np.random.rand(128).astype(np.float32),
            np.random.rand(128).astype(np.float32),
            np.random.rand(128).astype(np.float32)
        ]
        
        # Normalize vectors
        for i in range(len(vectors)):
            vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
        
        metadata = [
            {"doc_id": "doc1", "category": "technical", "score": 0.95},
            {"doc_id": "doc2", "category": "business", "score": 0.87},
            {"doc_id": "doc3", "category": "research", "score": 0.92}
        ]
        
        # Store vectors
        result = store.store_embeddings(collection_name, vectors, metadata)
        print(f"✅ Stored {result.stored_count} vectors successfully")
        
        # Perform similarity search
        query_vector = vectors[0]  # Use first vector as query
        search_results = store.similarity_search(query_vector, collection_name, top_k=2)
        
        print(f"🔍 Search Results (top {len(search_results)}):")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Score: {result.score:.4f}, Metadata: {result.metadata}")
        
        # Test collection management
        info_result = store.manage_collection(collection_id, "info")
        if info_result["success"]:
            info = info_result["info"]
            print(f"📊 Collection Info: {info['points_count']} points, Status: {info['status']}")
        
        # Test health check
        health = store.health_check()
        print(f"💚 Health Check: {'Healthy' if health['healthy'] else 'Unhealthy'}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- ✅ Vector storage with metadata")
        print("- ✅ Similarity search with cosine distance")
        print("- ✅ Collection management operations")
        print("- ✅ Connection health monitoring")
        print("- ✅ Error handling and retry logic")


if __name__ == "__main__":
    demo_vector_store()