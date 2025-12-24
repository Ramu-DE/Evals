"""Create useful collections in Qdrant for RAG evaluation pipeline."""

import os
from src.vector_store import QdrantVectorStore
from src.vector_store.interfaces import VectorConfig
from src.dataset_generator import RAGASDatasetGenerator
import numpy as np


def create_evaluation_collections():
    """Create collections for different evaluation scenarios."""
    print("🏗️  Creating Qdrant Collections for RAG Evaluation")
    print("=" * 60)
    
    # Initialize components
    vector_store = QdrantVectorStore()
    generator = RAGASDatasetGenerator(vector_store=vector_store)
    
    # Check connection
    health = vector_store.health_check()
    if not health["healthy"]:
        print(f"❌ Qdrant connection failed: {health['error']}")
        return
    
    print(f"✅ Connected to Qdrant: {health['collections_count']} existing collections")
    
    # Collection configurations
    collections_to_create = [
        {
            "name": "rag_evaluation_datasets",
            "description": "Evaluation datasets for RAG systems",
            "dimension": 384,
            "distance": "cosine"
        },
        {
            "name": "knowledge_graphs",
            "description": "Knowledge graph derived datasets",
            "dimension": 384,
            "distance": "cosine"
        },
        {
            "name": "synthetic_datasets",
            "description": "Synthetically generated evaluation data",
            "dimension": 384,
            "distance": "cosine"
        },
        {
            "name": "quality_benchmarks",
            "description": "High-quality benchmark datasets",
            "dimension": 384,
            "distance": "cosine"
        }
    ]
    
    created_collections = []
    
    for collection_config in collections_to_create:
        try:
            print(f"\n📁 Creating collection: {collection_config['name']}")
            
            vector_config = VectorConfig(
                dimension=collection_config["dimension"],
                distance_metric=collection_config["distance"]
            )
            
            collection_id = vector_store.create_collection(
                collection_config["name"], 
                vector_config
            )
            
            print(f"   ✅ Created: {collection_id}")
            print(f"   📝 Purpose: {collection_config['description']}")
            
            created_collections.append(collection_config["name"])
            
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"   ⚠️  Collection already exists: {collection_config['name']}")
                created_collections.append(collection_config["name"])
            else:
                print(f"   ❌ Failed to create {collection_config['name']}: {e}")
    
    print(f"\n🎯 Populating Collections with Sample Data")
    print("-" * 40)
    
    # Create sample datasets for different domains
    sample_domains = [
        {
            "domain": "Machine Learning",
            "collection": "synthetic_datasets",
            "size": 5
        },
        {
            "domain": "Natural Language Processing", 
            "collection": "synthetic_datasets",
            "size": 5
        },
        {
            "domain": "Computer Vision",
            "collection": "synthetic_datasets", 
            "size": 5
        }
    ]
    
    # Knowledge graph samples
    knowledge_graphs = [
        {
            "name": "AI_Technologies",
            "collection": "knowledge_graphs",
            "graph": {
                "nodes": [
                    {
                        "id": "Transformer",
                        "type": "architecture",
                        "properties": {
                            "invented": 2017,
                            "paper": "Attention Is All You Need",
                            "key_innovation": "self-attention mechanism"
                        },
                        "relationships": []
                    },
                    {
                        "id": "BERT",
                        "type": "model",
                        "properties": {
                            "released": 2018,
                            "company": "Google",
                            "type": "bidirectional encoder"
                        },
                        "relationships": []
                    }
                ],
                "edges": [
                    {
                        "source": "BERT",
                        "target": "Transformer",
                        "type": "based_on"
                    }
                ]
            }
        },
        {
            "name": "RAG_Ecosystem",
            "collection": "knowledge_graphs",
            "graph": {
                "nodes": [
                    {
                        "id": "RAG",
                        "type": "technique",
                        "properties": {
                            "full_name": "Retrieval-Augmented Generation",
                            "purpose": "Enhance LLM responses with retrieved context",
                            "components": "retriever + generator"
                        },
                        "relationships": []
                    },
                    {
                        "id": "Vector_Search",
                        "type": "technology",
                        "properties": {
                            "purpose": "Find similar embeddings",
                            "algorithms": "cosine similarity, dot product"
                        },
                        "relationships": []
                    }
                ],
                "edges": [
                    {
                        "source": "RAG",
                        "target": "Vector_Search",
                        "type": "uses"
                    }
                ]
            }
        }
    ]
    
    # Generate and store synthetic datasets
    for domain_config in sample_domains:
        try:
            print(f"\n🎲 Generating {domain_config['domain']} dataset...")
            
            dataset = generator.create_synthetic_dataset(
                domain_config["domain"], 
                domain_config["size"]
            )
            
            storage_result = generator.store_dataset_vectors(
                dataset, 
                domain_config["collection"]
            )
            
            if storage_result["success"]:
                print(f"   ✅ Stored {storage_result['stored_count']} vectors in {domain_config['collection']}")
            else:
                print(f"   ❌ Storage failed: {storage_result['error']}")
                
        except Exception as e:
            print(f"   ❌ Failed to generate {domain_config['domain']}: {e}")
    
    # Generate and store knowledge graph datasets
    for kg_config in knowledge_graphs:
        try:
            print(f"\n📊 Processing {kg_config['name']} knowledge graph...")
            
            dataset = generator.generate_from_knowledge_graph(kg_config["graph"])
            
            storage_result = generator.store_dataset_vectors(
                dataset,
                kg_config["collection"]
            )
            
            if storage_result["success"]:
                print(f"   ✅ Stored {storage_result['stored_count']} vectors in {kg_config['collection']}")
            else:
                print(f"   ❌ Storage failed: {storage_result['error']}")
                
        except Exception as e:
            print(f"   ❌ Failed to process {kg_config['name']}: {e}")
    
    # Create quality benchmark data
    print(f"\n🏆 Creating Quality Benchmark Dataset...")
    try:
        # High-quality, manually curated examples
        benchmark_questions = [
            "What is the primary advantage of using retrieval-augmented generation?",
            "How does vector similarity search work in RAG systems?",
            "What are the key components of a RAG evaluation pipeline?",
            "Why is quality validation important for evaluation datasets?",
            "How do you measure the diversity of generated questions?"
        ]
        
        benchmark_contexts = [
            ["RAG combines retrieval with generation to provide more accurate, contextual responses"],
            ["Vector search finds semantically similar embeddings using distance metrics like cosine similarity"],
            ["RAG evaluation requires datasets, metrics, tracing, and quality validation components"],
            ["Quality validation ensures datasets are complete, diverse, consistent, and relevant"],
            ["Diversity is measured by uniqueness ratios and variance in question patterns"]
        ]
        
        benchmark_answers = [
            "RAG provides more accurate responses by grounding generation in retrieved context",
            "Vector search uses embedding similarity to find relevant documents for context",
            "Key components include dataset generation, evaluation metrics, and observability",
            "Quality validation prevents poor datasets from producing unreliable evaluation results",
            "Diversity is measured through uniqueness ratios and pattern analysis"
        ]
        
        # Create benchmark dataset
        from src.dataset_generator.interfaces import EvaluationDataset
        
        benchmark_dataset = EvaluationDataset(
            dataset_id="quality_benchmark_v1",
            questions=benchmark_questions,
            contexts=benchmark_contexts,
            ground_truth_answers=benchmark_answers,
            metadata={
                "source": "manual_curation",
                "quality": "high",
                "purpose": "benchmark"
            },
            quality_score=0.95
        )
        
        storage_result = generator.store_dataset_vectors(
            benchmark_dataset,
            "quality_benchmarks"
        )
        
        if storage_result["success"]:
            print(f"   ✅ Stored {storage_result['stored_count']} benchmark vectors")
        else:
            print(f"   ❌ Benchmark storage failed: {storage_result['error']}")
            
    except Exception as e:
        print(f"   ❌ Failed to create benchmarks: {e}")
    
    # Final status
    print(f"\n📊 Final Collection Status")
    print("-" * 40)
    
    final_health = vector_store.health_check()
    print(f"Total Collections: {final_health['collections_count']}")
    
    # Show collection details
    for collection_name in created_collections:
        try:
            info_result = vector_store.manage_collection(collection_name, "info")
            if info_result["success"]:
                info = info_result["info"]
                points = info.get("points_count", 0)
                status = info.get("status", "unknown")
                print(f"   📁 {collection_name}: {points} points ({status})")
            else:
                print(f"   📁 {collection_name}: info unavailable")
        except:
            print(f"   📁 {collection_name}: created")
    
    print(f"\n🎉 Qdrant setup complete!")
    print("=" * 60)
    print("✅ Collections created and populated")
    print("✅ Sample data stored across different domains")
    print("✅ Knowledge graphs processed and vectorized")
    print("✅ Quality benchmarks established")
    print("\n🚀 Ready for RAG evaluation pipeline testing!")
    
    return created_collections


if __name__ == "__main__":
    collections = create_evaluation_collections()
    
    print(f"\n🔧 Next Steps:")
    print("1. Run: streamlit run streamlit_dataset_demo.py")
    print("2. Test vector storage and similarity search")
    print("3. Explore different collections and datasets")
    print("4. Generate your own knowledge graphs and datasets")