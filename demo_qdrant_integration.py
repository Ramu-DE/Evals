"""Demo showing Dataset Generator + Qdrant Vector Store integration."""

import os
from src.dataset_generator import RAGASDatasetGenerator
from src.vector_store import QdrantVectorStore
from src.vector_store.interfaces import VectorConfig


def demo_qdrant_integration():
    """Demonstrate Dataset Generator storing vectors in Qdrant."""
    print("🚀 RAG Evaluation Pipeline - Qdrant Integration Demo")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        print("⚠️  QDRANT_API_KEY not found in environment variables")
        print("   Set your Qdrant API key: export QDRANT_API_KEY=your_key_here")
        print("   Continuing with demo using mock operations...")
        vector_store = None
    else:
        print(f"✅ Found Qdrant API key: {api_key[:8]}...")
        
        # Initialize Qdrant vector store
        try:
            vector_store = QdrantVectorStore()
            health = vector_store.health_check()
            if health["healthy"]:
                print(f"✅ Connected to Qdrant: {health['collections_count']} collections")
            else:
                print(f"❌ Qdrant connection failed: {health['error']}")
                vector_store = None
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            vector_store = None
    
    # Initialize Dataset Generator with vector store
    generator = RAGASDatasetGenerator(vector_store=vector_store)
    
    print("\n📊 Step 1: Generate Dataset from Knowledge Graph")
    print("-" * 40)
    
    # Create sample knowledge graph
    sample_graph = {
        "nodes": [
            {
                "id": "RAG",
                "type": "ai_technique",
                "properties": {
                    "full_name": "Retrieval-Augmented Generation",
                    "purpose": "Enhance LLM responses with retrieved context",
                    "components": "retriever + generator"
                },
                "relationships": []
            },
            {
                "id": "Vector_Database",
                "type": "technology",
                "properties": {
                    "purpose": "Store and search high-dimensional vectors",
                    "examples": "Qdrant, Pinecone, Weaviate"
                },
                "relationships": []
            }
        ],
        "edges": [
            {
                "source": "RAG",
                "target": "Vector_Database",
                "type": "uses"
            }
        ]
    }
    
    # Generate dataset
    dataset = generator.generate_from_knowledge_graph(sample_graph)
    print(f"✅ Generated dataset: {len(dataset.questions)} questions")
    print(f"   Quality Score: {dataset.quality_score:.2f}")
    
    # Show sample questions
    print("\n📝 Sample Questions:")
    for i, (q, a) in enumerate(zip(dataset.questions[:2], dataset.ground_truth_answers[:2])):
        print(f"   {i+1}. Q: {q}")
        print(f"      A: {a}")
    
    print(f"\n🔗 Step 2: Store Dataset Vectors in Qdrant")
    print("-" * 40)
    
    if vector_store:
        # Store vectors in Qdrant
        storage_result = generator.store_dataset_vectors(
            dataset, 
            collection_name=f"demo_dataset_{dataset.dataset_id[:8]}"
        )
        
        if storage_result["success"]:
            print(f"✅ Stored {storage_result['stored_count']} vectors in Qdrant")
            print(f"   Collection: {storage_result['collection_name']}")
            
            # Test similarity search
            print(f"\n🔍 Step 3: Test Similarity Search")
            print("-" * 40)
            
            try:
                # Create a query vector (simple demo)
                query_text = "What is retrieval augmented generation?"
                query_vector = generator._text_to_vector(query_text)
                
                # Search for similar vectors
                search_results = vector_store.similarity_search(
                    query_vector=query_vector,
                    collection_name=storage_result['collection_name'],
                    top_k=3
                )
                
                print(f"Query: '{query_text}'")
                print(f"Found {len(search_results)} similar results:")
                
                for i, result in enumerate(search_results):
                    print(f"\n   Result {i+1} (Score: {result.score:.3f}):")
                    print(f"   Question: {result.metadata.get('question', 'N/A')}")
                    print(f"   Answer: {result.metadata.get('answer', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Similarity search failed: {e}")
        else:
            print(f"❌ Failed to store vectors: {storage_result['error']}")
    else:
        print("⚠️  Skipping vector storage (no Qdrant connection)")
        print("   In production, vectors would be stored with embeddings like:")
        print("   - Questions embedded using sentence-transformers")
        print("   - Contexts embedded for retrieval")
        print("   - Answers embedded for similarity matching")
    
    print(f"\n🎯 Step 4: Export Dataset in Multiple Formats")
    print("-" * 40)
    
    # Export in different formats
    formats = ["json", "jsonl"]
    for fmt in formats:
        export_result = generator.export_dataset(dataset, fmt)
        print(f"✅ Exported to {fmt.upper()}: {len(export_result['data'])} items")
    
    print(f"\n📈 Step 5: Quality Validation")
    print("-" * 40)
    
    quality_report = generator.validate_dataset_quality(dataset)
    print(f"Overall Quality: {quality_report.overall_score:.2f}")
    
    for metric, score in quality_report.quality_metrics.items():
        print(f"   {metric.capitalize()}: {score:.2f}")
    
    print(f"\n✨ Demo Complete!")
    print("=" * 60)
    print("Integration Features Demonstrated:")
    print("✓ Knowledge Graph → Dataset Generation")
    print("✓ Dataset → Vector Embeddings")
    if vector_store:
        print("✓ Vector Storage in Qdrant")
        print("✓ Similarity Search")
    else:
        print("⚠ Vector Storage (requires API key)")
    print("✓ Multi-format Export")
    print("✓ Quality Validation")
    
    return {
        "dataset": dataset,
        "vector_store": vector_store,
        "generator": generator
    }


if __name__ == "__main__":
    results = demo_qdrant_integration()
    
    if results["vector_store"]:
        print(f"\n🔧 To test with your own data:")
        print("1. Set QDRANT_API_KEY environment variable")
        print("2. Modify the knowledge graph in the demo")
        print("3. Run: python demo_qdrant_integration.py")
    else:
        print(f"\n🔧 To enable Qdrant integration:")
        print("1. Get your API key from Qdrant Cloud")
        print("2. Set: export QDRANT_API_KEY=your_key_here")
        print("3. Re-run this demo")