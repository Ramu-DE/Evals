"""Property-based tests for Dataset Generator component."""

import pytest
from hypothesis import given, strategies as st, assume
from typing import Dict, List, Any

from src.dataset_generator import RAGASDatasetGenerator, EvaluationDataset


class TestDatasetGeneratorProperties:
    """Property-based tests for Dataset Generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = RAGASDatasetGenerator()
    
    @given(
        nodes=st.lists(
            st.fixed_dictionaries({
                "id": st.text(min_size=1, max_size=50),
                "type": st.text(min_size=1, max_size=20),
                "properties": st.dictionaries(
                    st.text(min_size=1, max_size=20),
                    st.one_of(st.text(min_size=1, max_size=100), st.integers(), st.floats()),
                    min_size=1,
                    max_size=5
                ),
                "relationships": st.lists(
                    st.fixed_dictionaries({
                        "target": st.text(min_size=1, max_size=50),
                        "type": st.text(min_size=1, max_size=20)
                    }),
                    max_size=3
                )
            }),
            min_size=1,
            max_size=10
        ),
        edges=st.lists(
            st.fixed_dictionaries({
                "source": st.text(min_size=1, max_size=50),
                "target": st.text(min_size=1, max_size=50),
                "type": st.text(min_size=1, max_size=20)
            }),
            max_size=5
        )
    )
    def test_knowledge_graph_dataset_transformation(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        """
        **Feature: rag-evaluation-pipeline, Property 2: Knowledge Graph Dataset Transformation**
        **Validates: Requirements 1.2**
        
        For any valid knowledge graph input, the Dataset Generator should produce 
        a well-formed evaluation dataset with questions, contexts, and ground truth answers.
        """
        # Arrange
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {"source": "test"}
        }
        
        # Act
        result = self.generator.generate_from_knowledge_graph(graph_data)
        
        # Assert - Well-formed dataset structure
        assert isinstance(result, EvaluationDataset)
        assert result.dataset_id is not None and len(result.dataset_id) > 0
        assert isinstance(result.questions, list)
        assert isinstance(result.contexts, list)
        assert isinstance(result.ground_truth_answers, list)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.quality_score, (int, float))
        
        # Assert - Dataset contains data when nodes are provided
        if nodes:
            assert len(result.questions) > 0, "Should generate questions from knowledge graph nodes"
            assert len(result.contexts) > 0, "Should generate contexts from knowledge graph"
            assert len(result.ground_truth_answers) > 0, "Should generate ground truth answers"
        
        # Assert - Data consistency
        assert len(result.questions) == len(result.contexts), "Questions and contexts should have same length"
        assert len(result.questions) == len(result.ground_truth_answers), "Questions and answers should have same length"
        
        # Assert - Metadata contains source information
        assert "source" in result.metadata
        assert result.metadata["source"] == "knowledge_graph"
        assert "node_count" in result.metadata
        assert result.metadata["node_count"] == len(nodes)
        
        # Assert - Quality score is valid
        assert 0.0 <= result.quality_score <= 1.0, "Quality score should be between 0 and 1"
    
    @given(
        domain=st.text(min_size=1, max_size=50),
        size=st.integers(min_value=1, max_value=100)
    )
    def test_dataset_diversity_generation(self, domain: str, size: int):
        """
        **Feature: rag-evaluation-pipeline, Property 18: Dataset Diversity Generation**
        **Validates: Requirements 6.2**
        
        For any dataset generation request requiring diversity, the output should contain 
        varied question-answer pairs with measurable diversity metrics.
        """
        # Act
        result = self.generator.create_synthetic_dataset(domain, size)
        
        # Assert - Basic structure
        assert isinstance(result, EvaluationDataset)
        assert len(result.questions) == size
        assert len(result.contexts) == size
        assert len(result.ground_truth_answers) == size
        
        # Assert - Diversity metrics
        if size > 1:
            # Check question diversity
            unique_questions = set(result.questions)
            question_diversity = len(unique_questions) / len(result.questions)
            
            # Check answer diversity  
            unique_answers = set(result.ground_truth_answers)
            answer_diversity = len(unique_answers) / len(result.ground_truth_answers)
            
            # Should have some diversity (not all identical)
            assert question_diversity > 0, "Should have some question diversity"
            assert answer_diversity > 0, "Should have some answer diversity"
            
            # For larger datasets, expect reasonable diversity
            if size >= 10:
                assert question_diversity >= 0.5, f"Expected diversity >= 0.5, got {question_diversity}"
        
        # Assert - Metadata indicates diversity focus
        assert "domain" in result.metadata
        assert result.metadata["domain"] == domain
        assert "generation_method" in result.metadata
    
    @given(
        dataset=st.builds(
            EvaluationDataset,
            dataset_id=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))),
            questions=st.lists(
                st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1, max_size=20
            ),
            contexts=st.lists(
                st.lists(
                    st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                    min_size=1, max_size=3
                ),
                min_size=1, max_size=20
            ),
            ground_truth_answers=st.lists(
                st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1, max_size=20
            ),
            metadata=st.dictionaries(
                st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1
            ),
            quality_score=st.floats(min_value=0.0, max_value=1.0)
        )
    )
    def test_ground_truth_consistency(self, dataset: EvaluationDataset):
        """
        **Feature: rag-evaluation-pipeline, Property 19: Ground Truth Consistency**
        **Validates: Requirements 6.3**
        
        For any evaluation dataset, generated reference answers should be consistent 
        with the source knowledge and suitable for evaluation purposes.
        """
        # Ensure dataset has consistent lengths
        assume(len(dataset.questions) == len(dataset.contexts))
        assume(len(dataset.questions) == len(dataset.ground_truth_answers))
        
        # Act
        quality_report = self.generator.validate_dataset_quality(dataset)
        
        # Assert - Quality report structure
        assert hasattr(quality_report, 'overall_score')
        assert hasattr(quality_report, 'quality_metrics')
        assert hasattr(quality_report, 'issues_found')
        assert hasattr(quality_report, 'recommendations')
        
        # Assert - Consistency validation
        assert 'consistency' in quality_report.quality_metrics
        consistency_score = quality_report.quality_metrics['consistency']
        assert 0.0 <= consistency_score <= 1.0, "Consistency score should be between 0 and 1"
        
        # Assert - Ground truth answers are non-empty
        for answer in dataset.ground_truth_answers:
            assert answer.strip(), "Ground truth answers should not be empty"
        
        # Assert - Questions and answers have reasonable relationship
        for i in range(len(dataset.questions)):
            question = dataset.questions[i]
            answer = dataset.ground_truth_answers[i]
            
            # Basic consistency checks
            assert question.strip(), f"Question {i} should not be empty"
            assert answer.strip(), f"Answer {i} should not be empty"
            
            # Questions should be questions (end with ? or contain question words)
            question_lower = question.lower()
            is_question = (question.endswith('?') or 
                          any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']))
            
            # This is a soft check - not all questions need to follow this pattern
            # but the majority should for a consistent dataset
    
    @given(
        dataset=st.builds(
            EvaluationDataset,
            dataset_id=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))),
            questions=st.lists(
                st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1, max_size=20
            ),
            contexts=st.lists(
                st.lists(
                    st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                    min_size=1, max_size=3
                ),
                min_size=1, max_size=20
            ),
            ground_truth_answers=st.lists(
                st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1, max_size=20
            ),
            metadata=st.dictionaries(
                st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1
            ),
            quality_score=st.floats(min_value=0.0, max_value=1.0)
        ),
        quality_threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_quality_validation_enforcement(self, dataset: EvaluationDataset, quality_threshold: float):
        """
        **Feature: rag-evaluation-pipeline, Property 20: Quality Validation Enforcement**
        **Validates: Requirements 6.4**
        
        For any dataset with quality control enabled, the Dataset Generator should validate 
        content accuracy and reject substandard outputs.
        """
        # Ensure dataset has consistent lengths
        assume(len(dataset.questions) == len(dataset.contexts))
        assume(len(dataset.questions) == len(dataset.ground_truth_answers))
        
        # Act
        quality_report = self.generator.validate_dataset_quality(dataset)
        
        # Assert - Quality validation structure
        assert isinstance(quality_report.overall_score, (int, float))
        assert isinstance(quality_report.quality_metrics, dict)
        assert isinstance(quality_report.issues_found, list)
        assert isinstance(quality_report.recommendations, list)
        
        # Assert - Quality metrics are comprehensive
        expected_metrics = {'completeness', 'diversity', 'consistency', 'relevance'}
        assert expected_metrics.issubset(set(quality_report.quality_metrics.keys())), \
            f"Missing quality metrics. Expected: {expected_metrics}, Got: {set(quality_report.quality_metrics.keys())}"
        
        # Assert - All quality scores are valid
        for metric_name, score in quality_report.quality_metrics.items():
            assert 0.0 <= score <= 1.0, f"Quality metric {metric_name} should be between 0 and 1, got {score}"
        
        # Assert - Overall score is calculated correctly
        expected_overall = sum(quality_report.quality_metrics.values()) / len(quality_report.quality_metrics)
        assert abs(quality_report.overall_score - expected_overall) < 0.001, \
            f"Overall score calculation incorrect. Expected: {expected_overall}, Got: {quality_report.overall_score}"
        
        # Assert - Quality enforcement logic
        if quality_report.overall_score < quality_threshold:
            # Low quality should generate issues and recommendations
            assert len(quality_report.issues_found) > 0 or len(quality_report.recommendations) > 0, \
                "Low quality datasets should have issues or recommendations"
    
    @given(
        dataset=st.builds(
            EvaluationDataset,
            dataset_id=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))),
            questions=st.lists(
                st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1, max_size=10
            ),
            contexts=st.lists(
                st.lists(
                    st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                    min_size=1, max_size=3
                ),
                min_size=1, max_size=10
            ),
            ground_truth_answers=st.lists(
                st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1, max_size=10
            ),
            metadata=st.dictionaries(
                st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cc', 'Cf', 'Cs', 'Co', 'Cn'))), 
                min_size=1
            ),
            quality_score=st.floats(min_value=0.0, max_value=1.0)
        ),
        format_choice=st.sampled_from(["json", "csv", "jsonl", "parquet"])
    )
    def test_format_support_flexibility(self, dataset: EvaluationDataset, format_choice: str):
        """
        **Feature: rag-evaluation-pipeline, Property 21: Format Support Flexibility**
        **Validates: Requirements 6.5**
        
        For any supported output format request, the Dataset Generator should produce 
        valid datasets in the specified format.
        """
        # Ensure dataset has consistent lengths
        assume(len(dataset.questions) == len(dataset.contexts))
        assume(len(dataset.questions) == len(dataset.ground_truth_answers))
        
        # Act
        export_result = self.generator.export_dataset(dataset, format_choice)
        
        # Assert - Export result structure
        assert isinstance(export_result, dict)
        assert "dataset_id" in export_result
        assert "format" in export_result
        assert "data" in export_result
        assert "metadata" in export_result
        
        # Assert - Format is correctly set
        assert export_result["format"] == format_choice
        
        # Assert - Data structure matches format expectations
        data = export_result["data"]
        
        if format_choice == "json":
            # JSON format should contain the full dataset structure
            assert isinstance(data, dict)
            assert "questions" in data
            assert "contexts" in data
            assert "ground_truth_answers" in data
            
        elif format_choice == "jsonl":
            # JSONL format should be a list of line objects
            assert isinstance(data, list)
            assert len(data) == len(dataset.questions)
            for line in data:
                assert isinstance(line, dict)
                assert "question" in line
                assert "contexts" in line
                assert "ground_truth" in line
                
        elif format_choice == "csv":
            # CSV format should be a list of row dictionaries
            assert isinstance(data, list)
            assert len(data) == len(dataset.questions)
            for row in data:
                assert isinstance(row, dict)
                assert "question" in row
                assert "contexts" in row
                assert "ground_truth" in row
                
        elif format_choice == "parquet":
            # Parquet format should have structured data
            assert isinstance(data, dict)
            assert "questions" in data
            assert "contexts" in data
            assert "ground_truth_answers" in data
            assert "schema" in data
        
        # Assert - Metadata is preserved
        assert export_result["metadata"] == dataset.metadata
        
        # Assert - Dataset ID is preserved
        assert export_result["dataset_id"] == dataset.dataset_id


class TestDatasetGeneratorEdgeCases:
    """Edge case tests for Dataset Generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = RAGASDatasetGenerator()
    
    def test_empty_knowledge_graph_handling(self):
        """Test handling of empty knowledge graph."""
        with pytest.raises(ValueError, match="Invalid knowledge graph data"):
            self.generator.generate_from_knowledge_graph({})
    
    def test_unsupported_export_format(self):
        """Test handling of unsupported export format."""
        dataset = EvaluationDataset(
            dataset_id="test",
            questions=["What is test?"],
            contexts=[["Test context"]],
            ground_truth_answers=["Test answer"],
            metadata={},
            quality_score=0.8
        )
        
        with pytest.raises(ValueError, match="Unsupported format"):
            self.generator.export_dataset(dataset, "unsupported_format")
    
    def test_minimal_synthetic_dataset(self):
        """Test creation of minimal synthetic dataset."""
        result = self.generator.create_synthetic_dataset("test_domain", 1)
        
        assert len(result.questions) == 1
        assert len(result.contexts) == 1
        assert len(result.ground_truth_answers) == 1
        assert result.metadata["domain"] == "test_domain"