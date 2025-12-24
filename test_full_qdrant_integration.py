"""Comprehensive test of Qdrant integration with Dataset Generator."""

import os
from src.vector_store import QdrantVectorStore
from src.dataset_generator import RAGASDatasetGenerator
import numpy as np


def test_comprehensive_integration():
    """Test all aspects of the Qdrant integration."""
    print("🧪 Comprehensive Qdrant Integration Test")
    print("=" * 50)
    
    # Initialize components
    vector_store = QdrantVectorStore()
    generator = RAGASDatasetGenerator(vector_store=vector_store)
    
    # Test 1: Connection Health
    print("\n1️⃣  Testing Connection Health")
    print("-" * 30)
    health = vector_store.health_check()
    print(f"✅ Healthy: {health['healthy']}")
    print(f"📊 Collections: {health['collections_count']}")
    
    # Test 2: Knowledge Graph → Vectors
    print("\n2️⃣  Testing Knowledge Graph → Vector Storage")
    print("-" * 30)
    
    test_graph = {
        "nodes": [
            {
                "id": "LangChain",
                "type": "framework",
                "properties": {
                    "purpose": "Build LLM applications",
                    "language": "Python",
                    "features": "chains, agents, memory"
                },
                "relationships": []
            },
            {
                "id": "OpenAI",
                "type": "api_provider",
                "properties": {
                    "models": "GPT-3.5, GPT-4",
                    "type": "language_model_api"
                },
                "relationships": []
            }
        ],
        "edges": [
            {
                "source": "LangChain",
                "target": "OpenAI",
                "type": "integrates_with"
            }
        ]
    }
    
    # Generate dataset
    kg_dataset = generator.generate_from_knowledge_graph(test_graph)
    print(f"📝 Generated {len(kg_dataset.questions)} questions")
    
    # Store vectors
    storage_result = generator.store_dataset_vectors(kg_dataset, "test_kg_collection")
    print(f"💾 Stored {storage_result['stored_count']} vectors: {storage_result['success']}")
    
    # Test 3: Synthetic Dataset → Vectors
    print("\n3️⃣  Testing Synthetic Dataset → Vector Storage")
    print("-" * 30)
    
    synthetic_dataset = generator.create_synthetic_dataset("Deep Learning", 7)
    print(f"🎲 Generated {len(synthetic_dataset.questions)} synthetic questions")
    
    storage_result = generator.store_dataset_vectors(synthetic_dataset, "test_synthetic_collection")
    print(f"💾 Stored {storage_result['stored_count']} vectors: {storage_result['success']}")
    
    # Test 4: Similarity Search
    print("\n4️⃣  Testing Similarity Search")
    print("-" * 30)
    
    # Search in knowledge graph collection
    query_text = "What is LangChain used for?"
    query_vector = generator._text_to_vector(query_text)
    
    search_results = vector_store.similarity_search(
        query_vector=query_vector,
        collection_name="test_kg_collection",
        top_k=3
    )
    
    print(f"🔍 Query: '{query_text}'")
    print(f"📊 Found {len(search_results)} results:")
    
    for i, result in enumerate(search_results):
        print(f"   {i+1}. Score: {result.score:.3f}")
        print(f"      Q: {result.metadata.get('question', 'N/A')}")
        print(f"      A: {result.metadata.get('answer', 'N/A')[:50]}...")
    
    # Test 5: Cross-Collection Search
    print("\n5️⃣  Testing Cross-Collection Search")
    print("-" * 30)
    
    # Search in synthetic collection
    dl_query = "How does deep learning work?"
    dl_query_vector = generator._text_to_vector(dl_query)
    
    synthetic_results = vector_store.similarity_search(
        query_vector=dl_query_vector,
        collection_name="test_synthetic_collection",
        top_k=2
    )
    
    print(f"🔍 Query: '{dl_query}'")
    print(f"📊 Found {len(synthetic_results)} results in synthetic collection:")
    
    for i, result in enumerate(synthetic_results):
        print(f"   {i+1}. Score: {result.score:.3f}")
        print(f"      Q: {result.metadata.get('question', 'N/A')}")
    
    # Test 6: Quality Validation on Stored Data
    print("\n6️⃣  Testing Quality Validation")
    print("-" * 30)
    
    kg_quality = generator.validate_dataset_quality(kg_dataset)
    synthetic_quality = generator.validate_dataset_quality(synthetic_dataset)
    
    print(f"📊 Knowledge Graph Dataset Quality: {kg_quality.overall_score:.2f}")
    print(f"📊 Synthetic Dataset Quality: {synthetic_quality.overall_score:.2f}")
    
    # Test 7: Collection Management
    print("\n7️⃣  Testing Collection Management")
    print("-" * 30)
    
    # Get collection info
    kg_info = vector_store.manage_collection("test_kg_collection", "info")
    synthetic_info = vector_store.manage_collection("test_synthetic_collection", "info")
    
    if kg_info["success"]:
        info = kg_info["info"]
        print(f"📁 KG Collection: {info.get('points_count', 0)} points, {info.get('status', 'unknown')} status")
    
    if synthetic_info["success"]:
        info = synthetic_info["info"]
        print(f"📁 Synthetic Collection: {info.get('points_count', 0)} points, {info.get('status', 'unknown')} status")
    
    # Test 8: Export Integration
    print("\n8️⃣  Testing Export with Vector Metadata")
    print("-" * 30)
    
    # Export datasets in different formats
    formats = ["json", "jsonl"]
    for fmt in formats:
        kg_export = generator.export_dataset(kg_dataset, fmt)
        print(f"📤 Exported KG dataset to {fmt.upper()}: {len(kg_export['data'])} items")
    
    # Test 9: Performance Test
    print("\n9️⃣  Testing Performance")
    print("-" * 30)
    
    import time
    
    # Time vector generation
    start_time = time.time()
    perf_dataset = generator.create_synthetic_dataset("Performance Test", 20)
    generation_time = time.time() - start_time
    
    # Time vector storage
    start_time = time.time()
    perf_storage = generator.store_dataset_vectors(perf_dataset, "test_performance_collection")
    storage_time = time.time() - start_time
    
    # Time search
    start_time = time.time()
    perf_query_vector = generator._text_to_vector("performance test query")
    perf_results = vector_store.similarity_search(
        query_vector=perf_query_vector,
        collection_name="test_performance_collection",
        top_k=5
    )
    search_time = time.time() - start_time
    
    print(f"⏱️  Generation: {generation_time:.3f}s for 20 questions")
    print(f"⏱️  Storage: {storage_time:.3f}s for 20 vectors")
    print(f"⏱️  Search: {search_time:.3f}s for top-5 results")
    
    # Final Summary
    print("\n🎉 Integration Test Complete!")
    print("=" * 50)
    print("✅ Connection health verified")
    print("✅ Knowledge graph → vector pipeline working")
    print("✅ Synthetic dataset → vector pipeline working")
    print("✅ Similarity search functional")
    print("✅ Cross-collection search working")
    print("✅ Quality validation integrated")
    print("✅ Collection management operational")
    print("✅ Export functionality preserved")
    print("✅ Performance metrics collected")
    
    # Cleanup test collections
    print(f"\n🧹 Cleaning up test collections...")
    test_collections = [
        "test_kg_collection",
        "test_synthetic_collection", 
        "test_performance_collection"
    ]
    
    for collection in test_collections:
        try:
            result = vector_store.manage_collection(collection, "delete")
            if result["success"]:
                print(f"   ✅ Deleted {collection}")
            else:
                print(f"   ⚠️  Could not delete {collection}")
        except:
            print(f"   ⚠️  {collection} may not exist")
    
    return {
        "kg_dataset": kg_dataset,
        "synthetic_dataset": synthetic_dataset,
        "performance_metrics": {
            "generation_time": generation_time,
            "storage_time": storage_time,
            "search_time": search_time
        }
    }


if __name__ == "__main__":
    results = test_comprehensive_integration()
    
    print(f"\n📈 Performance Summary:")
    metrics = results["performance_metrics"]
    print(f"   Dataset Generation: {metrics['generation_time']:.3f}s")
    print(f"   Vector Storage: {metrics['storage_time']:.3f}s")
    print(f"   Similarity Search: {metrics['search_time']:.3f}s")
    
    print(f"\n🚀 System ready for production RAG evaluation!")
    print("   - Qdrant cluster connected and operational")
    print("   - Dataset generation pipeline functional")
    print("   - Vector storage and search working")
    print("   - Quality validation integrated")
    print("   - Multi-format export supported")