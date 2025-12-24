"""RAGAS-based evaluation engine implementation."""

import uuid
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

try:
    from ragas import evaluate
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from .interfaces import EvaluationEngine, EvaluationResult, TriadMetrics, EvaluationMode, ABTestResult, EvaluationReport


logger = logging.getLogger(__name__)


class RAGASEvaluationEngine(EvaluationEngine):
    """RAGAS-based implementation of the evaluation engine."""
    
    def __init__(self, trace_id_generator=None, use_mock=None):
        """Initialize the RAGAS evaluation engine.
        
        Args:
            trace_id_generator: Optional function to generate trace IDs
            use_mock: If True, use mock evaluation. If None, auto-detect based on API key availability
        """
        self.trace_id_generator = trace_id_generator or (lambda: str(uuid.uuid4()))
        
        # Determine if we should use mock evaluation
        if use_mock is None:
            # Auto-detect: use mock if no OpenAI API key is available or RAGAS is not available
            self.use_mock = not RAGAS_AVAILABLE or not os.getenv('OPENAI_API_KEY')
        else:
            self.use_mock = use_mock
            
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up RAGAS metrics for different evaluation modes."""
        if not self.use_mock and RAGAS_AVAILABLE:
            self.numerical_metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity
            ]
        else:
            # Mock metrics for testing
            self.numerical_metrics = [
                'faithfulness',
                'answer_relevancy', 
                'context_precision',
                'context_recall',
                'answer_correctness',
                'answer_similarity'
            ]
        
        self.binary_threshold = 0.7  # Threshold for binary evaluation
        
        # RAG-Triad metric mapping
        self.triad_metrics = {
            'retrieval_quality': ['context_precision', 'context_recall'],
            'generation_quality': ['faithfulness', 'answer_relevancy'],
            'overall_performance': ['answer_correctness', 'answer_similarity']
        }
    
    def evaluate_rag_system(self, query: str, context: List[str], response: str) -> EvaluationResult:
        """Evaluate a RAG system response using RAGAS metrics.
        
        Args:
            query: The input query
            context: List of retrieved context documents
            response: The generated response
            
        Returns:
            EvaluationResult with comprehensive metrics
        """
        evaluation_id = str(uuid.uuid4())
        trace_id = self.trace_id_generator()
        timestamp = datetime.now()
        
        try:
            if self.use_mock:
                # Mock evaluation for testing
                metrics = self._mock_evaluation(query, context, response)
            else:
                # Real RAGAS evaluation
                metrics = self._real_ragas_evaluation(query, context, response)
            
            # Generate binary results based on threshold
            binary_results = {
                metric: score >= self.binary_threshold 
                for metric, score in metrics.items()
            }
            
            # Calculate triad scores
            triad_scores = self._calculate_triad_scores(metrics)
            
            return EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=timestamp,
                metrics=metrics,
                binary_results=binary_results,
                triad_scores=triad_scores,
                metadata={
                    'query': query,
                    'context_count': len(context),
                    'response_length': len(response),
                    'evaluation_mode': EvaluationMode.NUMERICAL.value,
                    'mock_mode': self.use_mock
                },
                trace_id=trace_id
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for query '{query}': {str(e)}")
            # Return empty result with error metadata
            return EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=timestamp,
                metrics={},
                binary_results={},
                triad_scores=None,
                metadata={
                    'error': str(e),
                    'evaluation_mode': EvaluationMode.NUMERICAL.value,
                    'mock_mode': self.use_mock
                },
                trace_id=trace_id
            )
    
    def _mock_evaluation(self, query: str, context: List[str], response: str) -> Dict[str, float]:
        """Mock evaluation for testing purposes."""
        # Generate deterministic but realistic scores based on input characteristics
        import hashlib
        
        # Create a hash of the inputs for deterministic results
        input_hash = hashlib.md5(f"{query}{len(context)}{response}".encode()).hexdigest()
        hash_int = int(input_hash[:8], 16)
        
        # Generate scores between 0.3 and 0.9 based on hash
        base_score = 0.3 + (hash_int % 1000) / 1000 * 0.6
        
        metrics = {}
        for i, metric_name in enumerate(self.numerical_metrics):
            # Vary each metric slightly
            variation = (hash_int >> (i * 4)) % 100 / 1000  # 0-0.1 variation
            score = max(0.0, min(1.0, base_score + variation - 0.05))
            metrics[metric_name] = score
        
        return metrics
    
    def _real_ragas_evaluation(self, query: str, context: List[str], response: str) -> Dict[str, float]:
        """Real RAGAS evaluation using the RAGAS framework."""
        # Prepare dataset for RAGAS evaluation
        dataset_dict = {
            'question': [query],
            'contexts': [context],
            'answer': [response],
            'ground_truth': [response]  # Using response as ground truth for basic evaluation
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Run RAGAS evaluation
        result = evaluate(dataset, metrics=self.numerical_metrics)
        
        # Extract metrics
        metrics = {}
        for metric_name, score in result.items():
            if isinstance(score, list) and len(score) > 0:
                metrics[metric_name] = float(score[0])
            elif isinstance(score, (int, float)):
                metrics[metric_name] = float(score)
        
        return metrics
    
    def run_binary_evaluation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Run binary (pass/fail) evaluation on test cases.
        
        Args:
            test_cases: List of test cases with query, context, response, and expected result
            
        Returns:
            Dictionary mapping test case IDs to pass/fail results
        """
        results = {}
        
        for i, test_case in enumerate(test_cases):
            test_id = test_case.get('id', f'test_{i}')
            
            try:
                query = test_case['query']
                context = test_case['context']
                response = test_case['response']
                
                # Run evaluation
                eval_result = self.evaluate_rag_system(query, context, response)
                
                # Determine pass/fail based on average score
                avg_score = sum(eval_result.metrics.values()) / len(eval_result.metrics) if eval_result.metrics else 0
                results[test_id] = avg_score >= self.binary_threshold
                
            except Exception as e:
                logger.error(f"Binary evaluation failed for test case {test_id}: {str(e)}")
                results[test_id] = False
        
        return results
    
    def execute_rag_triad(self, dataset: List[Dict[str, Any]]) -> TriadMetrics:
        """Execute RAG-Triad evaluation covering retrieval, generation, and overall performance.
        
        Args:
            dataset: List of evaluation examples with query, context, response, ground_truth
            
        Returns:
            TriadMetrics with scores for each component
        """
        if not dataset:
            return TriadMetrics(
                retrieval_quality=0.0,
                generation_quality=0.0,
                overall_performance=0.0,
                component_scores={}
            )
        
        try:
            if self.use_mock:
                # Mock evaluation for testing
                component_scores = self._mock_triad_evaluation(dataset)
            else:
                # Real RAGAS evaluation
                component_scores = self._real_triad_evaluation(dataset)
            
            # Calculate component scores
            retrieval_scores = []
            generation_scores = []
            overall_scores = []
            
            for metric_name, avg_score in component_scores.items():
                # Categorize metrics
                if metric_name in ['context_precision', 'context_recall']:
                    retrieval_scores.append(avg_score)
                elif metric_name in ['faithfulness', 'answer_relevancy']:
                    generation_scores.append(avg_score)
                else:
                    overall_scores.append(avg_score)
            
            # Calculate triad averages
            retrieval_quality = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
            generation_quality = sum(generation_scores) / len(generation_scores) if generation_scores else 0.0
            overall_performance = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            
            return TriadMetrics(
                retrieval_quality=retrieval_quality,
                generation_quality=generation_quality,
                overall_performance=overall_performance,
                component_scores=component_scores
            )
            
        except Exception as e:
            logger.error(f"RAG-Triad evaluation failed: {str(e)}")
            return TriadMetrics(
                retrieval_quality=0.0,
                generation_quality=0.0,
                overall_performance=0.0,
                component_scores={'error': str(e)}
            )
    
    def _mock_triad_evaluation(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Mock triad evaluation for testing."""
        import hashlib
        
        # Generate deterministic scores based on dataset characteristics
        dataset_hash = hashlib.md5(str(len(dataset)).encode()).hexdigest()
        hash_int = int(dataset_hash[:8], 16)
        
        component_scores = {}
        for i, metric_name in enumerate(self.numerical_metrics):
            # Generate scores between 0.4 and 0.8
            base_score = 0.4 + (hash_int >> (i * 4)) % 100 / 100 * 0.4
            component_scores[metric_name] = base_score
        
        return component_scores
    
    def _real_triad_evaluation(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Real triad evaluation using RAGAS."""
        # Prepare dataset for RAGAS
        dataset_dict = {
            'question': [item['query'] for item in dataset],
            'contexts': [item['context'] for item in dataset],
            'answer': [item['response'] for item in dataset],
            'ground_truth': [item.get('ground_truth', item['response']) for item in dataset]
        }
        
        ragas_dataset = Dataset.from_dict(dataset_dict)
        
        # Evaluate with all metrics
        result = evaluate(ragas_dataset, metrics=self.numerical_metrics)
        
        # Calculate component scores
        component_scores = {}
        for metric_name, scores in result.items():
            if isinstance(scores, list):
                avg_score = sum(scores) / len(scores) if scores else 0.0
            else:
                avg_score = float(scores) if scores else 0.0
            
            component_scores[metric_name] = avg_score
        
        return component_scores
    
    def llm_as_judge_evaluation(self, responses: List[str], criteria: List[str]) -> Dict[str, Any]:
        """Execute LLM-as-a-Judge evaluation using language models to assess responses.
        
        Args:
            responses: List of responses to evaluate
            criteria: List of evaluation criteria
            
        Returns:
            Dictionary with judgment results and scores
        """
        # Note: This is a simplified implementation
        # In a full implementation, this would integrate with an LLM API
        # for actual judgment-based evaluation
        
        results = {
            'judgments': [],
            'scores': [],
            'criteria_scores': {},
            'overall_score': 0.0,
            'metadata': {
                'num_responses': len(responses),
                'num_criteria': len(criteria),
                'evaluation_mode': EvaluationMode.LLM_JUDGE.value
            }
        }
        
        try:
            # Simplified scoring based on response characteristics
            for i, response in enumerate(responses):
                response_score = self._simple_llm_judge_score(response, criteria)
                results['judgments'].append({
                    'response_id': i,
                    'score': response_score,
                    'criteria_met': response_score >= 0.7
                })
                results['scores'].append(response_score)
            
            # Calculate criteria-specific scores
            for criterion in criteria:
                criterion_scores = [
                    self._evaluate_criterion(response, criterion) 
                    for response in responses
                ]
                results['criteria_scores'][criterion] = {
                    'average': sum(criterion_scores) / len(criterion_scores),
                    'scores': criterion_scores
                }
            
            # Calculate overall score
            if results['scores']:
                results['overall_score'] = sum(results['scores']) / len(results['scores'])
            
            return results
            
        except Exception as e:
            logger.error(f"LLM-as-Judge evaluation failed: {str(e)}")
            results['metadata']['error'] = str(e)
            return results
    
    def _calculate_triad_scores(self, metrics: Dict[str, float]) -> TriadMetrics:
        """Calculate RAG-Triad scores from individual metrics."""
        retrieval_metrics = ['context_precision', 'context_recall']
        generation_metrics = ['faithfulness', 'answer_relevancy']
        overall_metrics = ['answer_correctness', 'answer_similarity']
        
        retrieval_scores = [metrics.get(m, 0.0) for m in retrieval_metrics if m in metrics]
        generation_scores = [metrics.get(m, 0.0) for m in generation_metrics if m in metrics]
        overall_scores = [metrics.get(m, 0.0) for m in overall_metrics if m in metrics]
        
        return TriadMetrics(
            retrieval_quality=sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0,
            generation_quality=sum(generation_scores) / len(generation_scores) if generation_scores else 0.0,
            overall_performance=sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
            component_scores=metrics
        )
    
    def _simple_llm_judge_score(self, response: str, criteria: List[str]) -> float:
        """Simple heuristic-based scoring for LLM judge evaluation."""
        # This is a placeholder implementation
        # In practice, this would call an actual LLM API
        
        score = 0.5  # Base score
        
        # Simple heuristics
        if len(response) > 50:  # Adequate length
            score += 0.1
        if len(response.split()) > 10:  # Sufficient word count
            score += 0.1
        if any(keyword in response.lower() for keyword in ['because', 'therefore', 'however']):
            score += 0.1  # Contains reasoning words
        if response.count('.') >= 2:  # Multiple sentences
            score += 0.1
        
        # Criteria-based adjustments
        for criterion in criteria:
            if criterion.lower() in response.lower():
                score += 0.05
        
        return min(1.0, score)  # Cap at 1.0
    
    def _evaluate_criterion(self, response: str, criterion: str) -> float:
        """Evaluate a single criterion for a response."""
        # Simplified criterion evaluation
        if criterion.lower() in response.lower():
            return 0.8
        elif any(word in response.lower() for word in criterion.lower().split()):
            return 0.6
        else:
            return 0.3
    
    def run_ab_test(self, system_a_results: List[EvaluationResult], system_b_results: List[EvaluationResult]) -> ABTestResult:
        """Run A/B testing comparison between two RAG systems.
        
        Args:
            system_a_results: Evaluation results from system A
            system_b_results: Evaluation results from system B
            
        Returns:
            ABTestResult with comparative analysis
        """
        test_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        try:
            # Aggregate metrics for both systems
            system_a_metrics = self._aggregate_metrics(system_a_results)
            system_b_metrics = self._aggregate_metrics(system_b_results)
            
            # Compare systems metric by metric
            comparison_results = {}
            statistical_significance = {}
            
            for metric_name in system_a_metrics.keys():
                if metric_name in system_b_metrics:
                    a_score = system_a_metrics[metric_name]
                    b_score = system_b_metrics[metric_name]
                    
                    # Determine winner (simple comparison)
                    if abs(a_score - b_score) < 0.05:  # Tie threshold
                        comparison_results[metric_name] = 'tie'
                    elif a_score > b_score:
                        comparison_results[metric_name] = 'A'
                    else:
                        comparison_results[metric_name] = 'B'
                    
                    # Mock statistical significance (in practice, would use proper statistical tests)
                    statistical_significance[metric_name] = abs(a_score - b_score) > 0.1
            
            # Determine overall winner
            a_wins = sum(1 for result in comparison_results.values() if result == 'A')
            b_wins = sum(1 for result in comparison_results.values() if result == 'B')
            ties = sum(1 for result in comparison_results.values() if result == 'tie')
            
            if a_wins > b_wins:
                overall_winner = 'A'
            elif b_wins > a_wins:
                overall_winner = 'B'
            else:
                overall_winner = 'tie'
            
            # Calculate confidence level (simplified)
            total_comparisons = len(comparison_results)
            significant_comparisons = sum(statistical_significance.values())
            confidence_level = significant_comparisons / total_comparisons if total_comparisons > 0 else 0.0
            
            return ABTestResult(
                test_id=test_id,
                timestamp=timestamp,
                system_a_metrics=system_a_metrics,
                system_b_metrics=system_b_metrics,
                comparison_results=comparison_results,
                statistical_significance=statistical_significance,
                overall_winner=overall_winner,
                confidence_level=confidence_level,
                metadata={
                    'system_a_samples': len(system_a_results),
                    'system_b_samples': len(system_b_results),
                    'total_metrics_compared': len(comparison_results),
                    'significant_differences': significant_comparisons
                }
            )
            
        except Exception as e:
            logger.error(f"A/B test failed: {str(e)}")
            return ABTestResult(
                test_id=test_id,
                timestamp=timestamp,
                system_a_metrics={},
                system_b_metrics={},
                comparison_results={},
                statistical_significance={},
                overall_winner='error',
                confidence_level=0.0,
                metadata={'error': str(e)}
            )
    
    def _aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Aggregate metrics from multiple evaluation results."""
        if not results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        # Calculate averages
        aggregated = {}
        for metric_name in all_metrics:
            values = [result.metrics.get(metric_name, 0.0) for result in results if metric_name in result.metrics]
            if values:
                aggregated[metric_name] = sum(values) / len(values)
            else:
                aggregated[metric_name] = 0.0
        
        return aggregated
    
    def generate_detailed_report(self, evaluation_results: List[EvaluationResult], criteria: List[str]) -> EvaluationReport:
        """Generate detailed evaluation report with criterion breakdowns.
        
        Args:
            evaluation_results: List of evaluation results to analyze
            criteria: List of evaluation criteria to include in breakdown
            
        Returns:
            EvaluationReport with detailed analysis and breakdowns
        """
        report_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        try:
            if not evaluation_results:
                return EvaluationReport(
                    report_id=report_id,
                    timestamp=timestamp,
                    overall_metrics={},
                    criterion_breakdowns={},
                    performance_summary={},
                    recommendations=[],
                    detailed_analysis={},
                    metadata={'error': 'No evaluation results provided'}
                )
            
            # Calculate overall metrics
            overall_metrics = self._aggregate_metrics(evaluation_results)
            
            # Generate criterion breakdowns
            criterion_breakdowns = {}
            for criterion in criteria:
                criterion_breakdowns[criterion] = self._analyze_criterion(evaluation_results, criterion)
            
            # Generate performance summary
            performance_summary = self._generate_performance_summary(overall_metrics, evaluation_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(overall_metrics, criterion_breakdowns)
            
            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(evaluation_results, overall_metrics)
            
            return EvaluationReport(
                report_id=report_id,
                timestamp=timestamp,
                overall_metrics=overall_metrics,
                criterion_breakdowns=criterion_breakdowns,
                performance_summary=performance_summary,
                recommendations=recommendations,
                detailed_analysis=detailed_analysis,
                metadata={
                    'num_evaluations': len(evaluation_results),
                    'num_criteria': len(criteria),
                    'report_type': 'detailed_evaluation'
                }
            )
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return EvaluationReport(
                report_id=report_id,
                timestamp=timestamp,
                overall_metrics={},
                criterion_breakdowns={},
                performance_summary={},
                recommendations=[],
                detailed_analysis={},
                metadata={'error': str(e)}
            )
    
    def check_performance_thresholds(self, evaluation_results: List[EvaluationResult], thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Check evaluation results against performance thresholds and flag issues.
        
        Args:
            evaluation_results: List of evaluation results to check
            thresholds: Dictionary mapping metric names to threshold values
            
        Returns:
            Dictionary with threshold check results and flagged issues
        """
        try:
            if not evaluation_results:
                return {
                    'flagged_issues': [],
                    'threshold_violations': {},
                    'performance_status': 'no_data',
                    'summary': 'No evaluation results to check',
                    'metadata': {'num_evaluations': 0, 'num_thresholds': len(thresholds)}
                }
            
            # Aggregate metrics
            overall_metrics = self._aggregate_metrics(evaluation_results)
            
            # Check thresholds
            threshold_violations = {}
            flagged_issues = []
            
            for metric_name, threshold_value in thresholds.items():
                if metric_name in overall_metrics:
                    actual_value = overall_metrics[metric_name]
                    
                    if actual_value < threshold_value:
                        violation = {
                            'metric': metric_name,
                            'threshold': threshold_value,
                            'actual': actual_value,
                            'severity': self._calculate_severity(actual_value, threshold_value),
                            'description': f"{metric_name} ({actual_value:.3f}) is below threshold ({threshold_value:.3f})"
                        }
                        threshold_violations[metric_name] = violation
                        flagged_issues.append(violation)
            
            # Determine overall performance status
            if not flagged_issues:
                performance_status = 'passing'
                summary = 'All metrics meet performance thresholds'
            elif len(flagged_issues) <= len(thresholds) * 0.3:  # Less than 30% violations
                performance_status = 'warning'
                summary = f'{len(flagged_issues)} metrics below threshold'
            else:
                performance_status = 'failing'
                summary = f'{len(flagged_issues)} metrics significantly below threshold'
            
            return {
                'flagged_issues': flagged_issues,
                'threshold_violations': threshold_violations,
                'performance_status': performance_status,
                'summary': summary,
                'overall_metrics': overall_metrics,
                'metadata': {
                    'num_evaluations': len(evaluation_results),
                    'num_thresholds': len(thresholds),
                    'num_violations': len(flagged_issues)
                }
            }
            
        except Exception as e:
            logger.error(f"Threshold checking failed: {str(e)}")
            return {
                'flagged_issues': [],
                'threshold_violations': {},
                'performance_status': 'error',
                'summary': f'Threshold checking failed: {str(e)}',
                'metadata': {'error': str(e)}
            }
    
    def _analyze_criterion(self, evaluation_results: List[EvaluationResult], criterion: str) -> Dict[str, Any]:
        """Analyze evaluation results for a specific criterion."""
        # Extract relevant metrics for this criterion
        criterion_metrics = []
        for result in evaluation_results:
            # Look for metrics that match or relate to this criterion
            matching_metrics = {
                name: value for name, value in result.metrics.items()
                if criterion.lower() in name.lower() or name.lower() in criterion.lower()
            }
            if matching_metrics:
                criterion_metrics.extend(matching_metrics.values())
        
        if not criterion_metrics:
            # If no direct matches, use all metrics as a fallback
            criterion_metrics = [
                value for result in evaluation_results 
                for value in result.metrics.values()
            ]
        
        if criterion_metrics:
            return {
                'average_score': sum(criterion_metrics) / len(criterion_metrics),
                'min_score': min(criterion_metrics),
                'max_score': max(criterion_metrics),
                'score_distribution': self._calculate_distribution(criterion_metrics),
                'sample_count': len(criterion_metrics),
                'pass_rate': sum(1 for score in criterion_metrics if score >= self.binary_threshold) / len(criterion_metrics)
            }
        else:
            return {
                'average_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'score_distribution': {},
                'sample_count': 0,
                'pass_rate': 0.0
            }
    
    def _generate_performance_summary(self, overall_metrics: Dict[str, float], evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate performance summary from metrics and results."""
        if not overall_metrics:
            return {
                'overall_score': 0.0,
                'grade': 'F',
                'strengths': [],
                'weaknesses': [],
                'trend': 'stable'
            }
        
        # Calculate overall score
        overall_score = sum(overall_metrics.values()) / len(overall_metrics)
        
        # Assign grade
        if overall_score >= 0.9:
            grade = 'A'
        elif overall_score >= 0.8:
            grade = 'B'
        elif overall_score >= 0.7:
            grade = 'C'
        elif overall_score >= 0.6:
            grade = 'D'
        else:
            grade = 'F'
        
        # Identify strengths and weaknesses
        strengths = [name for name, score in overall_metrics.items() if score >= 0.8]
        weaknesses = [name for name, score in overall_metrics.items() if score < 0.6]
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'trend': 'stable',  # Simplified - would need historical data for real trend analysis
            'total_evaluations': len(evaluation_results)
        }
    
    def _generate_recommendations(self, overall_metrics: Dict[str, float], criterion_breakdowns: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on metrics and criterion analysis."""
        recommendations = []
        
        # Check for low-performing metrics
        for metric_name, score in overall_metrics.items():
            if score < 0.6:
                recommendations.append(f"Improve {metric_name} performance (current: {score:.2f})")
            elif score < 0.7:
                recommendations.append(f"Monitor {metric_name} performance (current: {score:.2f})")
        
        # Check criterion-specific recommendations
        for criterion, analysis in criterion_breakdowns.items():
            if analysis.get('pass_rate', 0) < 0.7:
                recommendations.append(f"Focus on improving {criterion} (pass rate: {analysis.get('pass_rate', 0):.1%})")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance is good - continue current practices")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_detailed_analysis(self, evaluation_results: List[EvaluationResult], overall_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed analysis of evaluation results."""
        analysis = {
            'metric_correlations': {},
            'performance_patterns': {},
            'outlier_detection': {},
            'confidence_intervals': {}
        }
        
        try:
            # Simple correlation analysis
            metric_names = list(overall_metrics.keys())
            for i, metric1 in enumerate(metric_names):
                for metric2 in metric_names[i+1:]:
                    correlation = self._calculate_simple_correlation(evaluation_results, metric1, metric2)
                    analysis['metric_correlations'][f"{metric1}_vs_{metric2}"] = correlation
            
            # Performance patterns
            analysis['performance_patterns'] = {
                'consistent_performers': [name for name, score in overall_metrics.items() if score > 0.8],
                'variable_performers': [name for name, score in overall_metrics.items() if 0.6 <= score <= 0.8],
                'poor_performers': [name for name, score in overall_metrics.items() if score < 0.6]
            }
            
            # Simple outlier detection
            for metric_name in overall_metrics.keys():
                metric_values = [result.metrics.get(metric_name, 0) for result in evaluation_results if metric_name in result.metrics]
                if metric_values:
                    mean_val = sum(metric_values) / len(metric_values)
                    outliers = [val for val in metric_values if abs(val - mean_val) > 0.3]  # Simple threshold
                    analysis['outlier_detection'][metric_name] = {
                        'count': len(outliers),
                        'percentage': len(outliers) / len(metric_values) * 100
                    }
            
        except Exception as e:
            logger.warning(f"Detailed analysis generation failed: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate score distribution."""
        if not values:
            return {}
        
        distribution = {
            'excellent': sum(1 for v in values if v >= 0.9),
            'good': sum(1 for v in values if 0.8 <= v < 0.9),
            'fair': sum(1 for v in values if 0.7 <= v < 0.8),
            'poor': sum(1 for v in values if v < 0.7)
        }
        
        return distribution
    
    def _calculate_severity(self, actual: float, threshold: float) -> str:
        """Calculate severity of threshold violation."""
        difference = threshold - actual
        
        if difference <= 0.1:
            return 'low'
        elif difference <= 0.2:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_simple_correlation(self, evaluation_results: List[EvaluationResult], metric1: str, metric2: str) -> float:
        """Calculate simple correlation between two metrics."""
        values1 = [result.metrics.get(metric1, 0) for result in evaluation_results if metric1 in result.metrics]
        values2 = [result.metrics.get(metric2, 0) for result in evaluation_results if metric2 in result.metrics]
        
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0
        
        # Simple correlation calculation
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
        denominator1 = sum((v1 - mean1) ** 2 for v1 in values1) ** 0.5
        denominator2 = sum((v2 - mean2) ** 2 for v2 in values2) ** 0.5
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        return numerator / (denominator1 * denominator2)
    
    def aggregate_evaluation_results(self, evaluation_results: List[EvaluationResult], aggregation_method: str = 'mean') -> Dict[str, float]:
        """Aggregate multiple evaluation results using specified method.
        
        Args:
            evaluation_results: List of evaluation results to aggregate
            aggregation_method: Method to use ('mean', 'median', 'min', 'max')
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not evaluation_results:
            return {}
        
        try:
            # Collect all metric names
            all_metrics = set()
            for result in evaluation_results:
                all_metrics.update(result.metrics.keys())
            
            aggregated = {}
            
            for metric_name in all_metrics:
                values = [result.metrics.get(metric_name, 0.0) for result in evaluation_results if metric_name in result.metrics]
                
                if not values:
                    aggregated[metric_name] = 0.0
                    continue
                
                if aggregation_method == 'mean':
                    aggregated[metric_name] = sum(values) / len(values)
                elif aggregation_method == 'median':
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    if n % 2 == 0:
                        aggregated[metric_name] = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
                    else:
                        aggregated[metric_name] = sorted_values[n//2]
                elif aggregation_method == 'min':
                    aggregated[metric_name] = min(values)
                elif aggregation_method == 'max':
                    aggregated[metric_name] = max(values)
                else:
                    # Default to mean
                    aggregated[metric_name] = sum(values) / len(values)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            return {}
    
    def analyze_evaluation_trends(self, historical_results: List[List[EvaluationResult]], time_periods: List[str]) -> Dict[str, Any]:
        """Analyze trends in evaluation results over time.
        
        Args:
            historical_results: List of evaluation result lists for each time period
            time_periods: List of time period labels
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            if not historical_results or len(historical_results) != len(time_periods):
                return {
                    'trends': {},
                    'summary': 'Insufficient data for trend analysis',
                    'metadata': {
                        'periods_analyzed': 0,
                        'total_evaluations': 0
                    }
                }
            
            # Aggregate results for each time period
            period_aggregates = []
            for period_results in historical_results:
                if period_results:
                    aggregated = self.aggregate_evaluation_results(period_results, 'mean')
                    period_aggregates.append(aggregated)
                else:
                    period_aggregates.append({})
            
            # Analyze trends for each metric
            trends = {}
            all_metrics = set()
            for aggregate in period_aggregates:
                all_metrics.update(aggregate.keys())
            
            for metric_name in all_metrics:
                metric_values = []
                for aggregate in period_aggregates:
                    metric_values.append(aggregate.get(metric_name, 0.0))
                
                # Calculate trend direction
                if len(metric_values) >= 2:
                    trend_direction = self._calculate_trend_direction(metric_values)
                    trend_strength = self._calculate_trend_strength(metric_values)
                    
                    trends[metric_name] = {
                        'direction': trend_direction,
                        'strength': trend_strength,
                        'values': metric_values,
                        'change': metric_values[-1] - metric_values[0] if len(metric_values) >= 2 else 0.0,
                        'percent_change': ((metric_values[-1] - metric_values[0]) / metric_values[0] * 100) if metric_values[0] != 0 else 0.0
                    }
            
            # Generate summary
            improving_metrics = [name for name, trend in trends.items() if trend['direction'] == 'improving']
            declining_metrics = [name for name, trend in trends.items() if trend['direction'] == 'declining']
            stable_metrics = [name for name, trend in trends.items() if trend['direction'] == 'stable']
            
            summary = f"Analyzed {len(time_periods)} periods. "
            if improving_metrics:
                summary += f"{len(improving_metrics)} metrics improving. "
            if declining_metrics:
                summary += f"{len(declining_metrics)} metrics declining. "
            if stable_metrics:
                summary += f"{len(stable_metrics)} metrics stable."
            
            total_evaluations = sum(len(period_results) for period_results in historical_results)
            
            return {
                'trends': trends,
                'summary': summary,
                'improving_metrics': improving_metrics,
                'declining_metrics': declining_metrics,
                'stable_metrics': stable_metrics,
                'time_periods': time_periods,
                'metadata': {
                    'periods_analyzed': len(time_periods),
                    'total_evaluations': total_evaluations,
                    'metrics_analyzed': len(all_metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            return {
                'trends': {},
                'summary': f'Trend analysis failed: {str(e)}',
                'metadata': {'error': str(e)}
            }
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate the overall trend direction from a series of values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        # Determine direction based on slope
        if slope > 0.01:  # Threshold for improvement
            return 'improving'
        elif slope < -0.01:  # Threshold for decline
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_trend_strength(self, values: List[float]) -> str:
        """Calculate the strength of the trend."""
        if len(values) < 2:
            return 'none'
        
        # Calculate coefficient of variation as a measure of trend strength
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            return 'none'
        
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        cv = std_dev / mean_val
        
        # Classify strength based on coefficient of variation
        if cv < 0.1:
            return 'strong'
        elif cv < 0.2:
            return 'moderate'
        else:
            return 'weak'