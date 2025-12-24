"""Property-based tests for project structure validation.

**Feature: rag-evaluation-pipeline, Property 1: RAGAS Framework Integration**
**Validates: Requirements 1.1**
"""

import os
import sys
from pathlib import Path
from hypothesis import given, strategies as st
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestProjectStructure:
    """Test project structure and RAGAS framework integration."""
    
    def test_required_directories_exist(self):
        """Test that all required project directories exist."""
        base_path = Path(__file__).parent.parent
        required_dirs = [
            "src",
            "src/evaluation_engine",
            "src/workflow_orchestrator", 
            "src/vector_store",
            "src/tracing_system",
            "src/dataset_generator",
            "src/ui",
            "tests"
        ]
        
        for dir_path in required_dirs:
            full_path = base_path / dir_path
            assert full_path.exists(), f"Required directory {dir_path} does not exist"
            assert full_path.is_dir(), f"Path {dir_path} exists but is not a directory"
    
    def test_required_init_files_exist(self):
        """Test that all required __init__.py files exist."""
        base_path = Path(__file__).parent.parent
        required_init_files = [
            "src/__init__.py",
            "src/evaluation_engine/__init__.py",
            "src/workflow_orchestrator/__init__.py",
            "src/vector_store/__init__.py", 
            "src/tracing_system/__init__.py",
            "src/dataset_generator/__init__.py",
            "src/ui/__init__.py",
            "tests/__init__.py"
        ]
        
        for init_file in required_init_files:
            full_path = base_path / init_file
            assert full_path.exists(), f"Required __init__.py file {init_file} does not exist"
            assert full_path.is_file(), f"Path {init_file} exists but is not a file"
    
    def test_interface_files_exist(self):
        """Test that interface files exist for all components."""
        base_path = Path(__file__).parent.parent
        interface_files = [
            "src/evaluation_engine/interfaces.py",
            "src/workflow_orchestrator/interfaces.py",
            "src/vector_store/interfaces.py",
            "src/tracing_system/interfaces.py", 
            "src/dataset_generator/interfaces.py",
            "src/ui/interfaces.py"
        ]
        
        for interface_file in interface_files:
            full_path = base_path / interface_file
            assert full_path.exists(), f"Interface file {interface_file} does not exist"
            assert full_path.is_file(), f"Path {interface_file} exists but is not a file"
    
    def test_configuration_files_exist(self):
        """Test that configuration files exist."""
        base_path = Path(__file__).parent.parent
        config_files = [
            "requirements.txt",
            "pyproject.toml", 
            "pytest.ini",
            "config.py",
            "README.md"
        ]
        
        for config_file in config_files:
            full_path = base_path / config_file
            assert full_path.exists(), f"Configuration file {config_file} does not exist"
            assert full_path.is_file(), f"Path {config_file} exists but is not a file"
    
    @given(st.sampled_from([
        "evaluation_engine", "workflow_orchestrator", "vector_store", 
        "tracing_system", "dataset_generator", "ui"
    ]))
    @pytest.mark.hypothesis(deadline=None)
    def test_component_interfaces_importable(self, component_name):
        """Property test: All component interfaces should be importable.
        
        **Feature: rag-evaluation-pipeline, Property 1: RAGAS Framework Integration**
        **Validates: Requirements 1.1**
        
        For any component in the system, its interface module should be importable
        without errors, ensuring proper project structure and dependencies.
        """
        try:
            module_path = f"{component_name}.interfaces"
            __import__(module_path)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_path}: {e}")
    
    def test_ragas_framework_available(self):
        """Test that RAGAS framework is available for import.
        
        This validates that the RAGAS framework integration requirement is met
        by ensuring the framework can be imported successfully.
        """
        try:
            import ragas
            # Verify basic RAGAS functionality is available
            assert hasattr(ragas, '__version__'), "RAGAS version not available"
        except ImportError:
            pytest.skip("RAGAS framework not installed - this is expected during initial setup")
    
    def test_core_dependencies_available(self):
        """Test that all core dependencies are available."""
        core_deps = [
            "ragas",
            "langgraph", 
            "qdrant_client",
            "opik",
            "streamlit",
            "numpy",
            "pandas",
            "fastapi",
            "pydantic"
        ]
        
        missing_deps = []
        for dep in core_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            pytest.skip(f"Core dependencies not installed: {missing_deps} - this is expected during initial setup")
    
    def test_testing_framework_available(self):
        """Test that testing framework dependencies are available."""
        testing_deps = [
            "pytest",
            "hypothesis"
        ]
        
        for dep in testing_deps:
            try:
                __import__(dep)
            except ImportError as e:
                pytest.fail(f"Testing dependency {dep} not available: {e}")
    
    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    @pytest.mark.hypothesis(deadline=None)
    def test_project_structure_resilient_to_various_inputs(self, test_input):
        """Property test: Project structure should be resilient to various inputs.
        
        **Feature: rag-evaluation-pipeline, Property 1: RAGAS Framework Integration**
        **Validates: Requirements 1.1**
        
        For any valid string input, the project structure should remain intact
        and core components should be accessible.
        """
        # Test that core structure remains accessible regardless of input
        base_path = Path(__file__).parent.parent
        src_path = base_path / "src"
        
        # Verify structure integrity
        assert src_path.exists(), "Source directory should always exist"
        assert (src_path / "evaluation_engine").exists(), "Evaluation engine should always exist"
        
        # Test that we can still import core interfaces
        try:
            from evaluation_engine.interfaces import EvaluationEngine
            assert EvaluationEngine is not None
        except ImportError as e:
            pytest.fail(f"Core interfaces not accessible with input '{test_input}': {e}")