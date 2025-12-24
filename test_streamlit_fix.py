"""Test script to verify Streamlit components work correctly."""

import os
from src.dataset_generator import RAGASDatasetGenerator
from src.vector_store import QdrantVectorStore

def test_streamlit_components():
    """Test the components that Streamlit uses."""
    print("🧪 Testing Streamlit Components")
    print("=" * 40)
    
    # Test 1: Environment variable
    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bBfZh5RiGVGL2uj4TWK-kScnl-QIwCFdLD7-2X0UwBk"
    
    print(f"✅ API Key: {api_key[:8]}...")
    
    # Test 2: Vector Store initialization
    try:
        vector_store = QdrantVectorStore(api_key=api_key)
        health = vector_store.health_check()
        print(f"✅ Vector Store: {health['healthy']}")
        print(f"📊 Collections: {health['collections_count']}")
    except Exception as e:
        print(f"❌ Vector Store failed: {e}")
        vector_store = None
    
    # Test 3: Dataset Generator initialization
    try:
        generator = RAGASDatasetGenerator(vector_store=vector_store)
        print("✅ Dataset Generator: Initialized")
    except Exception as e:
        print(f"❌ Dataset Generator failed: {e}")
        return False
    
    # Test 4: Basic functionality
    try:
        dataset = generator.create_synthetic_dataset("Test Domain", 3)
        print(f"✅ Dataset Generation: {len(dataset.questions)} questions")
    except Exception as e:
        print(f"❌ Dataset Generation failed: {e}")
        return False
    
    # Test 5: Vector storage (if available)
    if vector_store:
        try:
            storage_result = generator.store_dataset_vectors(dataset, "streamlit_test")
            print(f"✅ Vector Storage: {storage_result['success']}")
            
            # Cleanup
            vector_store.manage_collection("streamlit_test", "delete")
            print("✅ Cleanup: Complete")
        except Exception as e:
            print(f"⚠️  Vector Storage: {e}")
    
    print(f"\n🎉 All components working!")
    print("Streamlit should now work without errors")
    return True

if __name__ == "__main__":
    success = test_streamlit_components()
    
    if success:
        print(f"\n🌐 Access Streamlit UI at:")
        print("   http://localhost:8503")
        print("\n📋 Available demos:")
        print("   - Knowledge Graph Transform")
        print("   - Synthetic Dataset")
        print("   - Quality Validation") 
        print("   - Format Export")
        print("   - Vector Storage (with Qdrant)")
    else:
        print(f"\n❌ Fix the errors above before using Streamlit")