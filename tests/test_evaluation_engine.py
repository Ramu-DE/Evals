"""Property-based tests for the Evaluation Engine."""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime
from typing import Dict, List, Any

from src.evaluation_engine import RAGASEvaluationEngine, EvaluationResult, TriadMetrics, ABTestResult, EvaluationReport


class TestEvaluationEngineProperties:
    """Property-based tests for evaluation engine correctness properties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RAGASEvaluationEngine(use_mock=True)  # Force mock mode for testing
    
    @given(
        test_cases=st.lists(
            st.fixed_dictionaries({
                'id': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=1, max_size=10),
                'query': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
                'context': st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=30), min_size=1, max_size=3),
                'response': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50)
            }),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_binary_evaluation_consistency(self, test_cases: List[Dict[str, Any]]):
        """
        **Feature: rag-evaluation-pipeline, Property 3: Binary Evaluation Consistency**
        **Validates: Requirements 1.3, 7.2**
        
        For any evaluation request in binary mode, the Evaluation Engine should return 
        only boolean pass/fail results without numerical scores.
        """
        # Run binary evaluation
        results = self.engine.run_binary_evaluation(test_cases)
        
        # Verify all results are boolean
        for test_id, result in results.items():
            assert isinstance(result, bool), f"Result for {test_id} should be boolean, got {type(result)}"
        
        # Verify we have results for all test cases
        expected_ids = {case.get('id', f'test_{i}') for i, case in enumerate(test_cases)}
        assert set(results.keys()) == expected_ids, "Should have results for all test cases"
        
        # Verify results are deterministic (run again and compare)
        results2 = self.engine.run_binary_evaluation(test_cases)
        assert results == results2, "Binary evaluation should be deterministic"
    
    @given(
        dataset=st.lists(
            st.fixed_dictionaries({
                'query': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
                'context': st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=30), min_size=1, max_size=3),
                'response': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
                'ground_truth': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50)
            }),
            min_size=1,
            max_size=3  # Keep small for performance
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_rag_triad_completeness(self, dataset: List[Dict[str, Any]]):
        """
        **Feature: rag-evaluation-pipeline, Property 4: RAG-Triad Completeness**
        **Validates: Requirements 1.4**
        
        For any RAG-Triad evaluation request, the output should contain assessments 
        for all three components: retrieval quality, generation quality, and overall system performance.
        """
        # Execute RAG-Triad evaluation
        triad_result = self.engine.execute_rag_triad(dataset)
        
        # Verify result is TriadMetrics instance
        assert isinstance(triad_result, TriadMetrics), "Should return TriadMetrics instance"
        
        # Verify all three components are present
        assert hasattr(triad_result, 'retrieval_quality'), "Should have retrieval_quality"
        assert hasattr(triad_result, 'generation_quality'), "Should have generation_quality"
        assert hasattr(triad_result, 'overall_performance'), "Should have overall_performance"
        assert hasattr(triad_result, 'component_scores'), "Should have component_scores"
        
        # Verify scores are numeric and in valid range [0, 1]
        assert isinstance(triad_result.retrieval_quality, (int, float)), "Retrieval quality should be numeric"
        assert isinstance(triad_result.generation_quality, (int, float)), "Generation quality should be numeric"
        assert isinstance(triad_result.overall_performance, (int, float)), "Overall performance should be numeric"
        
        assert 0 <= triad_result.retrieval_quality <= 1, "Retrieval quality should be in [0, 1]"
        assert 0 <= triad_result.generation_quality <= 1, "Generation quality should be in [0, 1]"
        assert 0 <= triad_result.overall_performance <= 1, "Overall performance should be in [0, 1]"
        
        # Verify component_scores is a dictionary
        assert isinstance(triad_result.component_scores, dict), "Component scores should be a dictionary"
    
    @given(
        responses=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
            min_size=1,
            max_size=3
        ),
        criteria=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=20),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_llm_judge_integration(self, responses: List[str], criteria: List[str]):
        """
        **Feature: rag-evaluation-pipeline, Property 5: LLM Judge Integration**
        **Validates: Requirements 1.5**
        
        For any LLM-as-a-Judge evaluation request, the system should invoke language model 
        assessment and return properly formatted judgment results.
        """
        # Execute LLM-as-Judge evaluation
        judge_result = self.engine.llm_as_judge_evaluation(responses, criteria)
        
        # Verify result structure
        assert isinstance(judge_result, dict), "Should return dictionary result"
        
        # Verify required keys are present
        required_keys = ['judgments', 'scores', 'criteria_scores', 'overall_score', 'metadata']
        for key in required_keys:
            assert key in judge_result, f"Should contain '{key}' in result"
        
        # Verify judgments structure
        judgments = judge_result['judgments']
        assert isinstance(judgments, list), "Judgments should be a list"
        assert len(judgments) == len(responses), "Should have judgment for each response"
        
        for judgment in judgments:
            assert isinstance(judgment, dict), "Each judgment should be a dictionary"
            assert 'response_id' in judgment, "Judgment should have response_id"
            assert 'score' in judgment, "Judgment should have score"
            assert isinstance(judgment['score'], (int, float)), "Score should be numeric"
            assert 0 <= judgment['score'] <= 1, "Score should be in [0, 1]"
        
        # Verify scores
        scores = judge_result['scores']
        assert isinstance(scores, list), "Scores should be a list"
        assert len(scores) == len(responses), "Should have score for each response"
        
        # Verify criteria scores
        criteria_scores = judge_result['criteria_scores']
        assert isinstance(criteria_scores, dict), "Criteria scores should be a dictionary"
        
        # Verify overall score
        overall_score = judge_result['overall_score']
        assert isinstance(overall_score, (int, float)), "Overall score should be numeric"
        assert 0 <= overall_score <= 1, "Overall score should be in [0, 1]"
        
        # Verify metadata
        metadata = judge_result['metadata']
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
        assert metadata['num_responses'] == len(responses), "Metadata should track response count"
        assert metadata['num_criteria'] == len(criteria), "Metadata should track criteria count"
    
    @given(
        query=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
        context=st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=30), min_size=1, max_size=3),
        response=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_numerical_metrics_provision(self, query: str, context: List[str], response: str):
        """
        **Feature: rag-evaluation-pipeline, Property 22: Numerical Metrics Provision**
        **Validates: Requirements 7.1**
        
        For any score-based evaluation request, the Evaluation Engine should return 
        numerical performance metrics within valid ranges.
        """
        # Execute evaluation
        result = self.engine.evaluate_rag_system(query, context, response)
        
        # Verify result structure
        assert isinstance(result, EvaluationResult), "Should return EvaluationResult instance"
        
        # Verify metrics are present and numerical
        metrics = result.metrics
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        
        # Verify all metric values are numerical and in valid range
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, (int, float)), f"Metric '{metric_name}' should be numerical"
            assert 0 <= metric_value <= 1, f"Metric '{metric_name}' should be in [0, 1], got {metric_value}"
        
        # Verify evaluation result has required fields
        assert isinstance(result.evaluation_id, str), "Should have string evaluation_id"
        assert isinstance(result.timestamp, datetime), "Should have datetime timestamp"
        assert isinstance(result.binary_results, dict), "Should have binary_results dict"
        assert isinstance(result.metadata, dict), "Should have metadata dict"
        assert isinstance(result.trace_id, str), "Should have string trace_id"
        
        # Verify binary results correspond to metrics
        binary_results = result.binary_results
        for metric_name in metrics.keys():
            if metric_name in binary_results:
                assert isinstance(binary_results[metric_name], bool), f"Binary result for '{metric_name}' should be boolean"
    
    @given(
        system_a_data=st.lists(
            st.fixed_dictionaries({
                'query': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
                'context': st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=30), min_size=1, max_size=3),
                'response': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50)
            }),
            min_size=1,
            max_size=3
        ),
        system_b_data=st.lists(
            st.fixed_dictionaries({
                'query': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
                'context': st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=30), min_size=1, max_size=3),
                'response': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50)
            }),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_ab_testing_capability(self, system_a_data: List[Dict[str, Any]], system_b_data: List[Dict[str, Any]]):
        """
        **Feature: rag-evaluation-pipeline, Property 23: A/B Testing Capability**
        **Validates: Requirements 7.3**
        
        For any comparative analysis request, the Evaluation Engine should support 
        simultaneous evaluation of different RAG configurations.
        """
        # Generate evaluation results for both systems
        system_a_results = []
        for data in system_a_data:
            result = self.engine.evaluate_rag_system(data['query'], data['context'], data['response'])
            system_a_results.append(result)
        
        system_b_results = []
        for data in system_b_data:
            result = self.engine.evaluate_rag_system(data['query'], data['context'], data['response'])
            system_b_results.append(result)
        
        # Run A/B test
        ab_result = self.engine.run_ab_test(system_a_results, system_b_results)
        
        # Verify result structure
        assert isinstance(ab_result, ABTestResult), "Should return ABTestResult instance"
        
        # Verify required fields are present
        assert isinstance(ab_result.test_id, str), "Should have string test_id"
        assert isinstance(ab_result.timestamp, datetime), "Should have datetime timestamp"
        assert isinstance(ab_result.system_a_metrics, dict), "Should have system_a_metrics dict"
        assert isinstance(ab_result.system_b_metrics, dict), "Should have system_b_metrics dict"
        assert isinstance(ab_result.comparison_results, dict), "Should have comparison_results dict"
        assert isinstance(ab_result.statistical_significance, dict), "Should have statistical_significance dict"
        assert isinstance(ab_result.overall_winner, str), "Should have string overall_winner"
        assert isinstance(ab_result.confidence_level, (int, float)), "Should have numeric confidence_level"
        assert isinstance(ab_result.metadata, dict), "Should have metadata dict"
        
        # Verify comparison results contain valid values
        valid_winners = {'A', 'B', 'tie', 'error'}
        assert ab_result.overall_winner in valid_winners, f"Overall winner should be one of {valid_winners}"
        
        for metric_name, winner in ab_result.comparison_results.items():
            assert winner in valid_winners, f"Winner for {metric_name} should be one of {valid_winners}"
        
        # Verify statistical significance values are boolean
        for metric_name, is_significant in ab_result.statistical_significance.items():
            assert isinstance(is_significant, bool), f"Statistical significance for {metric_name} should be boolean"
        
        # Verify confidence level is in valid range
        assert 0 <= ab_result.confidence_level <= 1, "Confidence level should be in [0, 1]"
        
        # Verify metrics are aggregated properly
        for metric_name, value in ab_result.system_a_metrics.items():
            assert isinstance(value, (int, float)), f"System A metric {metric_name} should be numeric"
            assert 0 <= value <= 1, f"System A metric {metric_name} should be in [0, 1]"
        
        for metric_name, value in ab_result.system_b_metrics.items():
            assert isinstance(value, (int, float)), f"System B metric {metric_name} should be numeric"
            assert 0 <= value <= 1, f"System B metric {metric_name} should be in [0, 1]"
        
        # Verify metadata contains expected information
        metadata = ab_result.metadata
        assert 'system_a_samples' in metadata, "Metadata should contain system_a_samples"
        assert 'system_b_samples' in metadata, "Metadata should contain system_b_samples"
        assert metadata['system_a_samples'] == len(system_a_results), "Should track correct number of A samples"
        assert metadata['system_b_samples'] == len(system_b_results), "Should track correct number of B samples"
    
    @given(
        evaluation_data=st.lists(
            st.fixed_dictionaries({
                'query': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
                'context': st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=30), min_size=1, max_size=3),
                'response': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50)
            }),
            min_size=1,
            max_size=3
        ),
        criteria=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=20),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_report_completeness(self, evaluation_data: List[Dict[str, Any]], criteria: List[str]):
        """
        **Feature: rag-evaluation-pipeline, Property 24: Report Completeness**
        **Validates: Requirements 7.4**
        
        For any evaluation report generation, the output should include detailed 
        breakdowns by all relevant evaluation criteria.
        """
        # Generate evaluation results
        evaluation_results = []
        for data in evaluation_data:
            result = self.engine.evaluate_rag_system(data['query'], data['context'], data['response'])
            evaluation_results.append(result)
        
        # Generate detailed report
        report = self.engine.generate_detailed_report(evaluation_results, criteria)
        
        # Verify report structure
        assert isinstance(report, EvaluationReport), "Should return EvaluationReport instance"
        
        # Verify required fields are present
        assert isinstance(report.report_id, str), "Should have string report_id"
        assert isinstance(report.timestamp, datetime), "Should have datetime timestamp"
        assert isinstance(report.overall_metrics, dict), "Should have overall_metrics dict"
        assert isinstance(report.criterion_breakdowns, dict), "Should have criterion_breakdowns dict"
        assert isinstance(report.performance_summary, dict), "Should have performance_summary dict"
        assert isinstance(report.recommendations, list), "Should have recommendations list"
        assert isinstance(report.detailed_analysis, dict), "Should have detailed_analysis dict"
        assert isinstance(report.metadata, dict), "Should have metadata dict"
        
        # Verify criterion breakdowns include all requested criteria
        for criterion in criteria:
            assert criterion in report.criterion_breakdowns, f"Should include breakdown for criterion '{criterion}'"
            
            breakdown = report.criterion_breakdowns[criterion]
            assert isinstance(breakdown, dict), f"Breakdown for '{criterion}' should be a dictionary"
            
            # Verify breakdown contains expected analysis fields
            expected_fields = ['average_score', 'min_score', 'max_score', 'sample_count', 'pass_rate']
            for field in expected_fields:
                assert field in breakdown, f"Breakdown for '{criterion}' should contain '{field}'"
                assert isinstance(breakdown[field], (int, float)), f"Field '{field}' should be numeric"
        
        # Verify performance summary contains required fields
        summary_fields = ['overall_score', 'grade', 'strengths', 'weaknesses']
        for field in summary_fields:
            assert field in report.performance_summary, f"Performance summary should contain '{field}'"
        
        # Verify overall score is in valid range
        overall_score = report.performance_summary['overall_score']
        assert 0 <= overall_score <= 1, "Overall score should be in [0, 1]"
        
        # Verify grade is valid
        valid_grades = ['A', 'B', 'C', 'D', 'F']
        assert report.performance_summary['grade'] in valid_grades, f"Grade should be one of {valid_grades}"
        
        # Verify recommendations are strings
        for recommendation in report.recommendations:
            assert isinstance(recommendation, str), "Each recommendation should be a string"
            assert len(recommendation) > 0, "Recommendations should not be empty"
        
        # Verify detailed analysis structure
        analysis_fields = ['metric_correlations', 'performance_patterns', 'outlier_detection']
        for field in analysis_fields:
            assert field in report.detailed_analysis, f"Detailed analysis should contain '{field}'"
        
        # Verify metadata contains expected information
        metadata = report.metadata
        assert 'num_evaluations' in metadata, "Metadata should contain num_evaluations"
        assert 'num_criteria' in metadata, "Metadata should contain num_criteria"
        assert metadata['num_evaluations'] == len(evaluation_results), "Should track correct number of evaluations"
        assert metadata['num_criteria'] == len(criteria), "Should track correct number of criteria"
    
    @given(
        evaluation_data=st.lists(
            st.fixed_dictionaries({
                'query': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50),
                'context': st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=5, max_size=30), min_size=1, max_size=3),
                'response': st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=50)
            }),
            min_size=1,
            max_size=3
        ),
        thresholds=st.dictionaries(
            keys=st.sampled_from(['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']),
            values=st.floats(min_value=0.1, max_value=0.9),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_threshold_based_flagging(self, evaluation_data: List[Dict[str, Any]], thresholds: Dict[str, float]):
        """
        **Feature: rag-evaluation-pipeline, Property 25: Threshold-Based Flagging**
        **Validates: Requirements 7.5**
        
        For any configured performance threshold, the Evaluation Engine should 
        automatically flag issues when metrics exceed acceptable limits.
        """
        # Generate evaluation results
        evaluation_results = []
        for data in evaluation_data:
            result = self.engine.evaluate_rag_system(data['query'], data['context'], data['response'])
            evaluation_results.append(result)
        
        # Check performance thresholds
        threshold_result = self.engine.check_performance_thresholds(evaluation_results, thresholds)
        
        # Verify result structure
        assert isinstance(threshold_result, dict), "Should return dictionary result"
        
        # Verify required fields are present
        required_fields = ['flagged_issues', 'threshold_violations', 'performance_status', 'summary', 'metadata']
        for field in required_fields:
            assert field in threshold_result, f"Should contain '{field}' in result"
        
        # Verify flagged issues structure
        flagged_issues = threshold_result['flagged_issues']
        assert isinstance(flagged_issues, list), "Flagged issues should be a list"
        
        for issue in flagged_issues:
            assert isinstance(issue, dict), "Each flagged issue should be a dictionary"
            issue_fields = ['metric', 'threshold', 'actual', 'severity', 'description']
            for field in issue_fields:
                assert field in issue, f"Flagged issue should contain '{field}'"
            
            # Verify severity is valid
            valid_severities = ['low', 'medium', 'high']
            assert issue['severity'] in valid_severities, f"Severity should be one of {valid_severities}"
            
            # Verify threshold violation logic
            assert issue['actual'] < issue['threshold'], "Flagged issue should have actual < threshold"
        
        # Verify threshold violations structure
        threshold_violations = threshold_result['threshold_violations']
        assert isinstance(threshold_violations, dict), "Threshold violations should be a dictionary"
        
        # Verify performance status is valid
        valid_statuses = ['passing', 'warning', 'failing', 'error', 'no_data']
        assert threshold_result['performance_status'] in valid_statuses, f"Performance status should be one of {valid_statuses}"
        
        # Verify summary is a string
        assert isinstance(threshold_result['summary'], str), "Summary should be a string"
        assert len(threshold_result['summary']) > 0, "Summary should not be empty"
        
        # Verify metadata contains expected information
        metadata = threshold_result['metadata']
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
        assert 'num_evaluations' in metadata, "Metadata should contain num_evaluations"
        assert 'num_thresholds' in metadata, "Metadata should contain num_thresholds"
        assert metadata['num_evaluations'] == len(evaluation_results), "Should track correct number of evaluations"
        assert metadata['num_thresholds'] == len(thresholds), "Should track correct number of thresholds"
        
        # Verify consistency between flagged issues and violations
        assert len(flagged_issues) == len(threshold_violations), "Flagged issues and violations should match"
        
        # Verify that all threshold violations are for metrics that were actually checked
        for metric_name in threshold_violations.keys():
            assert metric_name in thresholds, f"Violation for '{metric_name}' should correspond to a configured threshold"


class TestEvaluationEngineEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RAGASEvaluationEngine()
    
    def test_empty_dataset_rag_triad(self):
        """Test RAG-Triad evaluation with empty dataset."""
        result = self.engine.execute_rag_triad([])
        
        assert isinstance(result, TriadMetrics)
        assert result.retrieval_quality == 0.0
        assert result.generation_quality == 0.0
        assert result.overall_performance == 0.0
        assert isinstance(result.component_scores, dict)
    
    def test_empty_responses_llm_judge(self):
        """Test LLM judge evaluation with empty responses."""
        result = self.engine.llm_as_judge_evaluation([], ['criterion1'])
        
        assert isinstance(result, dict)
        assert result['judgments'] == []
        assert result['scores'] == []
        assert result['overall_score'] == 0.0
        assert result['metadata']['num_responses'] == 0
    
    def test_empty_criteria_llm_judge(self):
        """Test LLM judge evaluation with empty criteria."""
        result = self.engine.llm_as_judge_evaluation(['response1'], [])
        
        assert isinstance(result, dict)
        assert len(result['judgments']) == 1
        assert len(result['scores']) == 1
        assert result['metadata']['num_criteria'] == 0