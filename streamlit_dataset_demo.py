"""Streamlit UI demo for Dataset Generator."""

import streamlit as st
import json
import pandas as pd
import os
from src.dataset_generator import RAGASDatasetGenerator
from src.vector_store import QdrantVectorStore


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Dataset Generator Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Evaluation Pipeline - Dataset Generator")
    st.markdown("Interactive demo for testing dataset generation capabilities")
    
    # Initialize components
    if 'generator' not in st.session_state:
        # Check for Qdrant connection
        api_key = os.getenv("QDRANT_API_KEY")
        if not api_key:
            # Try to get from Streamlit secrets or manual input
            api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bBfZh5RiGVGL2uj4TWK-kScnl-QIwCFdLD7-2X0UwBk"
        
        vector_store = None
        
        if api_key:
            try:
                vector_store = QdrantVectorStore(api_key=api_key)
                health = vector_store.health_check()
                if not health["healthy"]:
                    vector_store = None
            except Exception as e:
                st.error(f"Qdrant connection failed: {e}")
                vector_store = None
        
        st.session_state.generator = RAGASDatasetGenerator(vector_store=vector_store)
        st.session_state.vector_store = vector_store
    
    # Show connection status
    with st.sidebar:
        st.title("🔗 Connection Status")
        if st.session_state.vector_store:
            st.success("✅ Qdrant Connected")
            health = st.session_state.vector_store.health_check()
            st.metric("Collections", health.get("collections_count", 0))
        else:
            st.warning("⚠️ Qdrant Disconnected")
            st.info("Set QDRANT_API_KEY environment variable to enable vector storage")
    
    # Sidebar for navigation
    st.sidebar.title("Demo Options")
    demo_type = st.sidebar.selectbox(
        "Choose Demo Type",
        ["Knowledge Graph Transform", "Synthetic Dataset", "Quality Validation", "Format Export", "Vector Storage"]
    )
    
    if demo_type == "Knowledge Graph Transform":
        knowledge_graph_demo()
    elif demo_type == "Synthetic Dataset":
        synthetic_dataset_demo()
    elif demo_type == "Quality Validation":
        quality_validation_demo()
    elif demo_type == "Format Export":
        format_export_demo()
    elif demo_type == "Vector Storage":
        vector_storage_demo()


def knowledge_graph_demo():
    """Knowledge graph transformation demo."""
    st.header("📊 Knowledge Graph Dataset Transformation")
    st.markdown("Transform knowledge graphs into evaluation datasets")
    
    # Sample knowledge graph input
    st.subheader("Input Knowledge Graph")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sample Knowledge Graph:**")
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
                    "relationships": []
                },
                {
                    "id": "Django",
                    "type": "web_framework", 
                    "properties": {
                        "type": "full-stack",
                        "architecture": "MVT"
                    },
                    "relationships": []
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
        
        # Allow user to edit the knowledge graph
        graph_json = st.text_area(
            "Edit Knowledge Graph (JSON):",
            value=json.dumps(sample_graph, indent=2),
            height=300
        )
    
    with col2:
        if st.button("🚀 Generate Dataset", type="primary"):
            try:
                # Parse the JSON
                graph_data = json.loads(graph_json)
                
                # Generate dataset
                with st.spinner("Generating dataset..."):
                    dataset = st.session_state.generator.generate_from_knowledge_graph(graph_data)
                
                # Display results
                st.success(f"✅ Generated dataset with {len(dataset.questions)} questions!")
                
                # Show metrics
                st.metric("Quality Score", f"{dataset.quality_score:.2f}")
                st.metric("Questions Generated", len(dataset.questions))
                
                # Show sample questions
                st.subheader("Generated Questions & Answers")
                for i, (q, c, a) in enumerate(zip(
                    dataset.questions[:3], 
                    dataset.contexts[:3], 
                    dataset.ground_truth_answers[:3]
                )):
                    with st.expander(f"Question {i+1}"):
                        st.write(f"**Q:** {q}")
                        st.write(f"**Context:** {c}")
                        st.write(f"**A:** {a}")
                
                # Store in session state for other demos
                st.session_state.last_dataset = dataset
                
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format. Please check your input.")
            except Exception as e:
                st.error(f"❌ Error generating dataset: {str(e)}")


def synthetic_dataset_demo():
    """Synthetic dataset generation demo."""
    st.header("🎲 Synthetic Dataset Generation")
    st.markdown("Generate synthetic evaluation datasets for any domain")
    
    col1, col2 = st.columns(2)
    
    with col1:
        domain = st.text_input("Domain", value="Machine Learning", help="Enter the domain for dataset generation")
        size = st.slider("Dataset Size", min_value=1, max_value=50, value=10, help="Number of question-answer pairs")
        
        if st.button("🎯 Generate Synthetic Dataset", type="primary"):
            with st.spinner(f"Generating {size} questions for {domain}..."):
                dataset = st.session_state.generator.create_synthetic_dataset(domain, size)
            
            st.success(f"✅ Generated synthetic dataset!")
            
            # Show metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Questions", len(dataset.questions))
            with col_b:
                st.metric("Quality Score", f"{dataset.quality_score:.2f}")
            with col_c:
                unique_q = len(set(dataset.questions))
                diversity = unique_q / len(dataset.questions)
                st.metric("Diversity", f"{diversity:.2f}")
            
            # Store in session state
            st.session_state.last_dataset = dataset
    
    with col2:
        if 'last_dataset' in st.session_state:
            st.subheader("Generated Dataset Preview")
            dataset = st.session_state.last_dataset
            
            # Create a dataframe for display
            df = pd.DataFrame({
                'Question': dataset.questions,
                'Answer': dataset.ground_truth_answers
            })
            
            st.dataframe(df, use_container_width=True)


def quality_validation_demo():
    """Quality validation demo."""
    st.header("🔍 Quality Validation")
    st.markdown("Validate dataset quality with comprehensive metrics")
    
    if 'last_dataset' not in st.session_state:
        st.warning("⚠️ Please generate a dataset first using one of the other demos.")
        return
    
    dataset = st.session_state.last_dataset
    
    if st.button("🔍 Validate Quality", type="primary"):
        with st.spinner("Validating dataset quality..."):
            quality_report = st.session_state.generator.validate_dataset_quality(dataset)
        
        # Display overall score
        st.metric("Overall Quality Score", f"{quality_report.overall_score:.2f}")
        
        # Display individual metrics
        st.subheader("Quality Metrics Breakdown")
        
        col1, col2, col3, col4 = st.columns(4)
        metrics = list(quality_report.quality_metrics.items())
        
        for i, (metric, score) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.metric(metric.capitalize(), f"{score:.2f}")
        
        # Show issues and recommendations
        if quality_report.issues_found:
            st.subheader("🚨 Issues Found")
            for issue in quality_report.issues_found:
                st.warning(f"• {issue}")
        
        if quality_report.recommendations:
            st.subheader("💡 Recommendations")
            for rec in quality_report.recommendations:
                st.info(f"• {rec}")
        
        # Quality visualization
        st.subheader("Quality Metrics Visualization")
        chart_data = pd.DataFrame({
            'Metric': list(quality_report.quality_metrics.keys()),
            'Score': list(quality_report.quality_metrics.values())
        })
        st.bar_chart(chart_data.set_index('Metric'))


def format_export_demo():
    """Format export demo."""
    st.header("📤 Format Export")
    st.markdown("Export datasets in multiple formats")
    
    if 'last_dataset' not in st.session_state:
        st.warning("⚠️ Please generate a dataset first using one of the other demos.")
        return
    
    dataset = st.session_state.last_dataset
    
    # Format selection
    format_choice = st.selectbox(
        "Choose Export Format",
        ["json", "jsonl", "csv", "parquet"],
        help="Select the format for dataset export"
    )
    
    if st.button("📤 Export Dataset", type="primary"):
        try:
            with st.spinner(f"Exporting to {format_choice.upper()}..."):
                export_result = st.session_state.generator.export_dataset(dataset, format_choice)
            
            st.success(f"✅ Dataset exported to {format_choice.upper()} format!")
            
            # Show export metadata
            st.subheader("Export Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Format:** {export_result['format']}")
                st.write(f"**Dataset ID:** {export_result['dataset_id']}")
            with col2:
                st.write(f"**Export ID:** {export_result['export_timestamp']}")
            
            # Show data preview
            st.subheader("Data Preview")
            data = export_result['data']
            
            if format_choice == "json":
                st.json(data)
            elif format_choice == "jsonl":
                st.write("JSONL Lines:")
                for i, line in enumerate(data[:3]):  # Show first 3 lines
                    st.code(json.dumps(line), language="json")
                    if i < len(data) - 1:
                        st.write("---")
            elif format_choice == "csv":
                df = pd.DataFrame(data)
                st.dataframe(df)
            elif format_choice == "parquet":
                st.write("Parquet Structure:")
                st.json({k: f"{len(v)} items" if isinstance(v, list) else v for k, v in data.items()})
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")


def vector_storage_demo():
    """Vector storage demo with Qdrant integration."""
    st.header("🔗 Vector Storage in Qdrant")
    st.markdown("Store and search dataset vectors in Qdrant vector database")
    
    if not st.session_state.vector_store:
        st.error("❌ Qdrant connection not available")
        st.info("To enable vector storage:")
        st.code("export QDRANT_API_KEY=your_api_key_here")
        return
    
    if 'last_dataset' not in st.session_state:
        st.warning("⚠️ Please generate a dataset first using one of the other demos.")
        return
    
    dataset = st.session_state.last_dataset
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Store Vectors")
        
        collection_name = st.text_input(
            "Collection Name",
            value=f"demo_dataset_{dataset.dataset_id[:8]}",
            help="Name for the Qdrant collection"
        )
        
        if st.button("🚀 Store in Qdrant", type="primary"):
            with st.spinner("Storing vectors in Qdrant..."):
                storage_result = st.session_state.generator.store_dataset_vectors(
                    dataset, collection_name
                )
            
            if storage_result["success"]:
                st.success(f"✅ Stored {storage_result['stored_count']} vectors!")
                st.session_state.last_collection = collection_name
                
                # Show storage details
                st.info(f"Collection: `{storage_result['collection_name']}`")
                
            else:
                st.error(f"❌ Storage failed: {storage_result['error']}")
    
    with col2:
        st.subheader("🔍 Similarity Search")
        
        if 'last_collection' in st.session_state:
            query_text = st.text_input(
                "Search Query",
                value="What is retrieval augmented generation?",
                help="Enter a question to search for similar content"
            )
            
            top_k = st.slider("Number of Results", 1, 10, 3)
            
            if st.button("🔍 Search", type="secondary"):
                try:
                    with st.spinner("Searching vectors..."):
                        # Create query vector
                        query_vector = st.session_state.generator._text_to_vector(query_text)
                        
                        # Search
                        results = st.session_state.vector_store.similarity_search(
                            query_vector=query_vector,
                            collection_name=st.session_state.last_collection,
                            top_k=top_k
                        )
                    
                    st.success(f"Found {len(results)} results")
                    
                    # Display results
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} (Score: {result.score:.3f})"):
                            st.write(f"**Question:** {result.metadata.get('question', 'N/A')}")
                            st.write(f"**Answer:** {result.metadata.get('answer', 'N/A')}")
                            st.write(f"**Dataset ID:** {result.metadata.get('dataset_id', 'N/A')}")
                
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
        else:
            st.info("Store vectors first to enable similarity search")
    
    # Vector storage statistics
    if st.session_state.vector_store:
        st.subheader("📊 Qdrant Statistics")
        
        try:
            health = st.session_state.vector_store.health_check()
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Connection", "✅ Healthy" if health["healthy"] else "❌ Unhealthy")
            with col_b:
                st.metric("Collections", health.get("collections_count", 0))
            with col_c:
                st.metric("URL", "Connected")
            
            # Show collection info if available
            if 'last_collection' in st.session_state:
                try:
                    collection_info = st.session_state.vector_store.manage_collection(
                        st.session_state.last_collection, "info"
                    )
                    
                    if collection_info["success"]:
                        info = collection_info["info"]
                        st.subheader(f"Collection: {st.session_state.last_collection}")
                        
                        col_x, col_y, col_z = st.columns(3)
                        with col_x:
                            st.metric("Status", info.get("status", "Unknown"))
                        with col_y:
                            st.metric("Points", info.get("points_count", 0))
                        with col_z:
                            st.metric("Vectors", info.get("vectors_count", 0))
                
                except Exception as e:
                    st.warning(f"Could not fetch collection info: {e}")
        
        except Exception as e:
            st.error(f"Could not fetch Qdrant statistics: {e}")


if __name__ == "__main__":
    main()