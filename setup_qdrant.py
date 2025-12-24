"""Setup script to test Qdrant connection."""

import os
from src.vector_store import QdrantVectorStore

def test_qdrant_connection():
    """Test Qdrant connection and setup."""
    print("🔧 Qdrant Connection Setup")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not api_key:
        print("❌ QDRANT_API_KEY not found")
        print("\n📋 Setup Instructions:")
        print("1. Go to https://cloud.qdrant.io/")
        print("2. Create account or login")
        print("3. Get your API key")
        print("4. Set environment variable:")
        print("   Windows: set QDRANT_API_KEY=your_key_here")
        print("   Linux/Mac: export QDRANT_API_KEY=your_key_here")
        print("5. Re-run this script")
        return False
    
    print(f"✅ Found API key: {api_key[:8]}...")
    
    try:
        # Test connection
        vector_store = QdrantVectorStore()
        health = vector_store.health_check()
        
        if health["healthy"]:
            print(f"✅ Connected to Qdrant successfully!")
            print(f"   Collections: {health['collections_count']}")
            print(f"   URL: {health['url']}")
            
            # Test basic operations
            from src.vector_store.interfaces import VectorConfig
            import numpy as np
            
            # Create test collection
            test_collection = "test_connection"
            vector_config = VectorConfig(
                dimension=384, 
                distance_metric="cosine",
                index_type="hnsw"
            )
            
            try:
                collection_id = vector_store.create_collection(test_collection, vector_config)
                print(f"✅ Created test collection: {collection_id}")
                
                # Test vector storage
                test_vectors = [np.random.rand(384) for _ in range(3)]
                test_metadata = [{"text": f"test_{i}", "index": i} for i in range(3)]
                
                storage_result = vector_store.store_embeddings(
                    collection_name=test_collection,
                    vectors=test_vectors,
                    metadata=test_metadata
                )
                
                if storage_result.success:
                    print(f"✅ Stored {storage_result.stored_count} test vectors")
                    
                    # Test search
                    query_vector = np.random.rand(384)
                    search_results = vector_store.similarity_search(
                        query_vector=query_vector,
                        collection_name=test_collection,
                        top_k=2
                    )
                    
                    print(f"✅ Search returned {len(search_results)} results")
                    
                    # Cleanup
                    vector_store.manage_collection(test_collection, "delete")
                    print(f"✅ Cleaned up test collection")
                    
                    print(f"\n🎉 Qdrant integration fully working!")
                    return True
                else:
                    print(f"❌ Vector storage failed: {storage_result.error_message}")
                    return False
                    
            except Exception as e:
                print(f"❌ Collection operations failed: {e}")
                return False
                
        else:
            print(f"❌ Connection failed: {health['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    
    if success:
        print(f"\n🚀 Ready to use Qdrant integration!")
        print("   Run: python demo_qdrant_integration.py")
        print("   Or: streamlit run streamlit_dataset_demo.py")
    else:
        print(f"\n🔧 Fix the connection issues above and try again")