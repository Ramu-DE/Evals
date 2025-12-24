# RAG Evaluation Pipeline

A comprehensive evaluation system for Retrieval-Augmented Generation (RAG) workflows that provides monitoring, tracing, and evaluation capabilities.

## 🚀 Features

- **Multi-methodology Evaluation**: Support for RAGAS, binary evaluation, RAG-Triad, and LLM-as-a-Judge
- **Workflow Orchestration**: LangGraph-based orchestration for complex evaluation workflows
- **Vector Storage**: Qdrant integration for efficient similarity search and retrieval
- **Observability**: OPIK integration for comprehensive tracing and monitoring
- **Dataset Generation**: Automated creation of evaluation datasets from knowledge graphs
- **User Interface**: Streamlit-based dashboard for visualization and interaction

## 📁 Project Structure

```
src/
├── evaluation_engine/     # Core evaluation logic using RAGAS
├── workflow_orchestrator/ # LangGraph-based workflow management
├── vector_store/         # Qdrant integration for vector operations
├── tracing_system/       # OPIK integration for observability
├── dataset_generator/    # Automated dataset creation
├── integration_layer/    # Component integration and coordination
└── ui/                   # Streamlit user interface

tests/                    # Comprehensive test suite
demos/                    # Example usage and demonstrations
.kiro/specs/             # Feature specifications and requirements
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ramu-DE/Evals.git
cd Evals
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rag-eval python=3.8
conda activate rag-eval
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```bash
# Required API Keys
QDRANT_API_KEY=your_qdrant_api_key_here
OPIK_API_KEY=your_opik_api_key_here
OPIK_WORKSPACE=your_opik_workspace_name

# Optional: OpenAI API Key for LLM evaluations
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_URL=https://your-cluster-url.qdrant.tech:6333
QDRANT_COLLECTION_NAME=rag_evaluation
```

### Step 5: Initialize Qdrant Collections

```bash
python create_qdrant_collections.py
```

### Step 6: Verify Installation

```bash
# Run basic tests
pytest tests/test_project_structure.py

# Run a demo
python demo_integration_layer.py
```

## 🚦 Quick Start

### Running the Streamlit Dashboard

```bash
streamlit run streamlit_dataset_demo.py
```

### Basic Usage Example

```python
from src.integration_layer import IntegrationManager
from src.evaluation_engine import RagasEvaluationEngine

# Initialize the system
manager = IntegrationManager()
evaluator = RagasEvaluationEngine()

# Run evaluation
results = evaluator.evaluate(
    questions=["What is RAG?"],
    contexts=[["RAG stands for Retrieval-Augmented Generation..."]],
    answers=["RAG is a technique that combines retrieval and generation..."]
)

print(f"Evaluation Results: {results}")
```

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage Report

```bash
pytest --cov=src --cov-report=html
```

### Run Property-Based Tests

```bash
pytest tests/ -v -k "property"
```

### Run Specific Component Tests

```bash
# Test evaluation engine
pytest tests/test_evaluation_engine.py

# Test vector store
pytest tests/test_vector_store.py

# Test integration layer
pytest tests/test_integration_layer.py
```

## 📊 Demo Scripts

The project includes several demonstration scripts:

- `demo_integration_layer.py` - Shows component integration
- `demo_evaluation_engine.py` - Demonstrates evaluation capabilities
- `demo_vector_store.py` - Vector storage operations
- `demo_workflow_orchestrator.py` - Workflow orchestration
- `demo_dataset_generator.py` - Dataset generation
- `demo_tracing_system.py` - Observability features

Run any demo:
```bash
python demo_<component_name>.py
```

## 🔧 Development

### Code Quality Tools

This project uses:
- **pytest** for unit testing
- **Hypothesis** for property-based testing
- **Black** for code formatting
- **MyPy** for type checking
- **Flake8** for linting

### Pre-commit Setup

```bash
pip install pre-commit
pre-commit install
```

### Running Code Quality Checks

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## 📚 Documentation

- [Requirements Specification](.kiro/specs/rag-evaluation-pipeline/requirements.md)
- [Design Document](.kiro/specs/rag-evaluation-pipeline/design.md)
- [Implementation Tasks](.kiro/specs/rag-evaluation-pipeline/tasks.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Verify your `QDRANT_API_KEY` and `QDRANT_URL` in `.env`
   - Check if Qdrant collections are created: `python create_qdrant_collections.py`

2. **OPIK Tracing Issues**
   - Ensure `OPIK_API_KEY` and `OPIK_WORKSPACE` are set correctly
   - Check OPIK service status

3. **Import Errors**
   - Verify virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **Test Failures**
   - Check if all environment variables are set
   - Run tests individually to isolate issues

### Getting Help

- Open an issue on GitHub
- Check the [documentation](.kiro/specs/)
- Review the demo scripts for usage examples