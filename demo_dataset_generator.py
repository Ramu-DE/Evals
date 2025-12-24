"""Interactive demo for the Dataset Generator component."""

import json
from src.dataset_generator import RAGASDatasetGenerator


def demo_knowledge_graph_transformation():
    """Demo knowledge graph to dataset transformation."""
    print("=" * 60)
    print("DEMO: Knowledge Graph Dataset Transformation")
    print("=" * 60)
    
    # Create sample knowledge graph data
    sample_graph = {
        "nodes": [
            {
                "id": "Python",
                "type": "programming_language",
                "properties": {
                    "creator": "Guido van Rossum",
                    "year": 1991,
                    "paradigm": "multi-paradigm"
                },
                "relationships": [
                    {"target": "Django", "type": "supports_framework"},
                    {"target": "Flask", "type": "supports_framework"}
                ]
            },
            {
                "id": "Django",
                "type": "web_framework",
                "properties": {
                    "type": "full-stack",
                    "architecture": "MVT"
                },
                "relationships": [
                    {"target": "Python", "type": "built_with"}
                ]
            }
        ],
        "edges": [
            {
                "source": "Python",
                "target": "Django",
                "type": "supports"
            }
        ]
    }
    
    generator = RAGASDatasetGenerator()
    dataset = generator.generate_from_knowledge_graph(sample_graph)
    
    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Generated {len(dataset.questions)} questions")
    print(f"Quality Score: {dataset.quality_score:.2f}")
    print("\nSample Questions and Answers:")
    
    for i, (question, contexts, answer) in enumerate(zip(
        dataset.questions[:3], 
        dataset.contexts[:3], 
        dataset.ground_truth_answers[:3]
    )):
        print(f"\n{i+1}. Question: {question}")
        print(f"   Context: {contexts}")
        print(f"   Answer: {answer}")
    
    return dataset


def demo_synthetic_dataset_generation():
    """Demo synthetic dataset generation."""
    print("\n" + "=" * 60)
    print("DEMO: Synthetic Dataset Generation")
    print("=" * 60)
    
    generator = RAGASDatasetGenerator()
    dataset = generator.create_synthetic_dataset("Machine Learning", 8)
    
    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Domain: {dataset.metadata['domain']}")
    print(f"Generated {len(dataset.questions)} questions")
    print(f"Quality Score: {dataset.quality_score:.2f}")
    
    # Show diversity
    unique_questions = set(dataset.questions)
    diversity = len(unique_questions) / len(dataset.questions)
    print(f"Question Diversity: {diversity:.2f}")
    
    print("\nGenerated Questions:")
    for i, (question, answer) in enumerate(zip(dataset.questions, dataset.ground_truth_answers)):
        print(f"{i+1}. Q: {question}")
        print(f"   A: {answer}")
    
    return dataset


def demo_quality_validation():
    """Demo quality validation."""
    print("\n" + "=" * 60)
    print("DEMO: Quality Validation")
    print("=" * 60)
    
    generator = RAGASDatasetGenerator()
    
    # Create a test dataset
    dataset = generator.create_synthetic_dataset("Data Science", 5)
    
    # Validate quality
    quality_report = generator.validate_dataset_quality(dataset)
    
    print(f"Overall Quality Score: {quality_report.overall_score:.2f}")
    print("\nQuality Metrics:")
    for metric, score in quality_report.quality_metrics.items():
        print(f"  {metric.capitalize()}: {score:.2f}")
    
    if quality_report.issues_found:
        print("\nIssues Found:")
        for issue in quality_report.issues_found:
            print(f"  - {issue}")
    
    if quality_report.recommendations:
        print("\nRecommendations:")
        for rec in quality_report.recommendations:
            print(f"  - {rec}")
    
    return quality_report


def demo_format_export():
    """Demo different export formats."""
    print("\n" + "=" * 60)
    print("DEMO: Format Export")
    print("=" * 60)
    
    generator = RAGASDatasetGenerator()
    dataset = generator.create_synthetic_dataset("AI Ethics", 3)
    
    formats = ["json", "jsonl", "csv", "parquet"]
    
    for format_type in formats:
        print(f"\n--- {format_type.upper()} Format ---")
        export_result = generator.export_dataset(dataset, format_type)
        
        print(f"Format: {export_result['format']}")
        print(f"Dataset ID: {export_result['dataset_id']}")
        
        # Show a sample of the data structure
        data = export_result['data']
        if format_type == "json":
            print(f"Questions count: {len(data['questions'])}")
            print(f"First question: {data['questions'][0]}")
        elif format_type == "jsonl":
            print(f"Lines count: {len(data)}")
            print(f"First line: {data[0]}")
        elif format_type == "csv":
            print(f"Rows count: {len(data)}")
            print(f"First row keys: {list(data[0].keys())}")
        elif format_type == "parquet":
            print(f"Schema: {data['schema']}")
            print(f"Questions count: {len(data['questions'])}")


def interactive_demo():
    """Run interactive demo."""
    print("🚀 RAG Evaluation Pipeline - Dataset Generator Demo")
    print("This demo showcases the Dataset Generator capabilities")
    
    try:
        # Run all demos
        kg_dataset = demo_knowledge_graph_transformation()
        synthetic_dataset = demo_synthetic_dataset_generation()
        quality_report = demo_quality_validation()
        demo_format_export()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("All Dataset Generator features demonstrated:")
        print("✓ Knowledge Graph Transformation")
        print("✓ Synthetic Dataset Generation")
        print("✓ Quality Validation")
        print("✓ Multi-format Export")
        
        return {
            "knowledge_graph_dataset": kg_dataset,
            "synthetic_dataset": synthetic_dataset,
            "quality_report": quality_report
        }
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = interactive_demo()
    
    if results:
        print(f"\n📊 Summary:")
        print(f"Knowledge Graph Dataset: {len(results['knowledge_graph_dataset'].questions)} questions")
        print(f"Synthetic Dataset: {len(results['synthetic_dataset'].questions)} questions")
        print(f"Average Quality Score: {(results['knowledge_graph_dataset'].quality_score + results['synthetic_dataset'].quality_score) / 2:.2f}")