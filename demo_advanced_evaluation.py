#!/usr/bin/env python3
"""
Demo script showcasing advanced evaluation capabilities.

This script demonstrates:
1. A/B testing between two RAG systems
2. Detailed evaluation reporting with criterion breakdowns
3. Threshold-based performance flagging and alerting
4. Evaluation result aggregation and trend analysis
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append('src')

from evaluation_engine import RAGASEvaluationEngine, EvaluationResult


def create_sample_evaluation_data() -> List[Dict[str, Any]]:
    """Create sample evaluation data for demonstration."""
    return [
        {
            'query': 'What is machine learning?',
            'context': ['Machine learning is a subset of artificial intelligence', 'It involves training algorithms on data'],
            'response': 'Machine learning is a method of data analysis that automates analytical model building.'
        },
        {
            'query': 'How does neural network work?',
            'context': ['Neural networks are computing systems inspired by biological neural networks', 'They consist of interconnected nodes'],
            'response': 'Neural networks work by processing information through interconnected nodes that simulate neurons.'
        },
        {
            'query': 'What is deep learning?',
            'context': ['Deep learning is a subset of machine learning', 'It uses neural networks with multiple layers'],
            'response': 'Deep learning uses multi-layered neural networks to learn complex patterns in data.'
        }
    ]


def demo_ab_testing():
    """Demonstrate A/B testing capabilities."""
    print("=" * 60)
    print("DEMO: A/B Testing Comparison")
    print("=" * 60)
    
    engine = RAGASEvaluationEngine(use_mock=True)
    sample_data = create_sample_evaluation_data()
    
    # Generate results for System A (simulate better performance)
    print("Generating evaluation results for System A...")
    system_a_results = []
    for data in sample_data:
        result = engine.evaluate_rag_system(data['query'], data['context'], data['response'])
        system_a_results.append(result)
    
    # Generate results for System B (simulate slightly worse performance)
    print("Generating evaluation results for System B...")
    system_b_results = []
    for data in sample_data:
        # Modify responses slightly to simulate different system
        modified_response = data['response'] + " This is system B."
        result = engine.evaluate_rag_system(data['query'], data['context'], modified_response)
        system_b_results.append(result)
    
    # Run A/B test
    print("Running A/B test comparison...")
    ab_result = engine.run_ab_test(system_a_results, system_b_results)
    
    print(f"\nA/B Test Results:")
    print(f"Test ID: {ab_result.test_id}")
    print(f"Overall Winner: {ab_result.overall_winner}")
    print(f"Confidence Level: {ab_result.confidence_level:.2%}")
    
    print(f"\nSystem A Metrics:")
    for metric, value in ab_result.system_a_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nSystem B Metrics:")
    for metric, value in ab_result.system_b_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nComparison Results:")
    for metric, winner in ab_result.comparison_results.items():
        significance = ab_result.statistical_significance.get(metric, False)
        sig_text = " (significant)" if significance else " (not significant)"
        print(f"  {metric}: {winner}{sig_text}")


def demo_detailed_reporting():
    """Demonstrate detailed evaluation reporting."""
    print("\n" + "=" * 60)
    print("DEMO: Detailed Evaluation Reporting")
    print("=" * 60)
    
    engine = RAGASEvaluationEngine(use_mock=True)
    sample_data = create_sample_evaluation_data()
    
    # Generate evaluation results
    print("Generating evaluation results...")
    evaluation_results = []
    for data in sample_data:
        result = engine.evaluate_rag_system(data['query'], data['context'], data['response'])
        evaluation_results.append(result)
    
    # Define evaluation criteria
    criteria = ['accuracy', 'relevance', 'completeness', 'clarity']
    
    # Generate detailed report
    print("Generating detailed evaluation report...")
    report = engine.generate_detailed_report(evaluation_results, criteria)
    
    print(f"\nDetailed Evaluation Report:")
    print(f"Report ID: {report.report_id}")
    print(f"Generated: {report.timestamp}")
    
    print(f"\nOverall Metrics:")
    for metric, value in report.overall_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nPerformance Summary:")
    summary = report.performance_summary
    print(f"  Overall Score: {summary.get('overall_score', 0):.3f}")
    print(f"  Grade: {summary.get('grade', 'N/A')}")
    print(f"  Strengths: {', '.join(summary.get('strengths', []))}")
    print(f"  Weaknesses: {', '.join(summary.get('weaknesses', []))}")
    
    print(f"\nCriterion Breakdowns:")
    for criterion, breakdown in report.criterion_breakdowns.items():
        print(f"  {criterion}:")
        print(f"    Average Score: {breakdown.get('average_score', 0):.3f}")
        print(f"    Pass Rate: {breakdown.get('pass_rate', 0):.1%}")
        print(f"    Sample Count: {breakdown.get('sample_count', 0)}")
    
    print(f"\nRecommendations:")
    for i, recommendation in enumerate(report.recommendations, 1):
        print(f"  {i}. {recommendation}")


def demo_threshold_flagging():
    """Demonstrate threshold-based performance flagging."""
    print("\n" + "=" * 60)
    print("DEMO: Threshold-Based Performance Flagging")
    print("=" * 60)
    
    engine = RAGASEvaluationEngine(use_mock=True)
    sample_data = create_sample_evaluation_data()
    
    # Generate evaluation results
    print("Generating evaluation results...")
    evaluation_results = []
    for data in sample_data:
        result = engine.evaluate_rag_system(data['query'], data['context'], data['response'])
        evaluation_results.append(result)
    
    # Define performance thresholds
    thresholds = {
        'faithfulness': 0.8,
        'answer_relevancy': 0.75,
        'context_precision': 0.7,
        'context_recall': 0.65
    }
    
    # Check performance thresholds
    print("Checking performance against thresholds...")
    threshold_result = engine.check_performance_thresholds(evaluation_results, thresholds)
    
    print(f"\nThreshold Check Results:")
    print(f"Performance Status: {threshold_result['performance_status']}")
    print(f"Summary: {threshold_result['summary']}")
    
    print(f"\nConfigured Thresholds:")
    for metric, threshold in thresholds.items():
        print(f"  {metric}: {threshold:.3f}")
    
    if threshold_result['flagged_issues']:
        print(f"\nFlagged Issues:")
        for issue in threshold_result['flagged_issues']:
            print(f"  ⚠️  {issue['metric']}: {issue['actual']:.3f} < {issue['threshold']:.3f} ({issue['severity']} severity)")
            print(f"      {issue['description']}")
    else:
        print(f"\n✅ No performance issues detected - all metrics meet thresholds!")
    
    print(f"\nThreshold Violations: {len(threshold_result['threshold_violations'])}")


def demo_trend_analysis():
    """Demonstrate evaluation trend analysis."""
    print("\n" + "=" * 60)
    print("DEMO: Evaluation Trend Analysis")
    print("=" * 60)
    
    engine = RAGASEvaluationEngine(use_mock=True)
    sample_data = create_sample_evaluation_data()
    
    # Simulate historical evaluation results over multiple time periods
    print("Generating historical evaluation data...")
    time_periods = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    historical_results = []
    
    for week in range(4):
        week_results = []
        for data in sample_data:
            # Simulate slight improvement over time by modifying responses
            modified_response = data['response'] + f" Enhanced in week {week + 1}."
            result = engine.evaluate_rag_system(data['query'], data['context'], modified_response)
            week_results.append(result)
        historical_results.append(week_results)
    
    # Analyze trends
    print("Analyzing evaluation trends...")
    trend_analysis = engine.analyze_evaluation_trends(historical_results, time_periods)
    
    print(f"\nTrend Analysis Results:")
    print(f"Summary: {trend_analysis['summary']}")
    
    print(f"\nMetric Trends:")
    for metric, trend in trend_analysis['trends'].items():
        direction = trend['direction']
        strength = trend['strength']
        change = trend['change']
        percent_change = trend['percent_change']
        
        direction_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}.get(direction, '❓')
        
        print(f"  {direction_emoji} {metric}:")
        print(f"    Direction: {direction} ({strength} strength)")
        print(f"    Change: {change:+.3f} ({percent_change:+.1f}%)")
        print(f"    Values: {[f'{v:.3f}' for v in trend['values']]}")
    
    if trend_analysis.get('improving_metrics'):
        print(f"\n📈 Improving Metrics: {', '.join(trend_analysis['improving_metrics'])}")
    
    if trend_analysis.get('declining_metrics'):
        print(f"\n📉 Declining Metrics: {', '.join(trend_analysis['declining_metrics'])}")
    
    if trend_analysis.get('stable_metrics'):
        print(f"\n➡️  Stable Metrics: {', '.join(trend_analysis['stable_metrics'])}")


def demo_aggregation_methods():
    """Demonstrate different aggregation methods."""
    print("\n" + "=" * 60)
    print("DEMO: Evaluation Result Aggregation")
    print("=" * 60)
    
    engine = RAGASEvaluationEngine(use_mock=True)
    sample_data = create_sample_evaluation_data()
    
    # Generate evaluation results
    print("Generating evaluation results...")
    evaluation_results = []
    for data in sample_data:
        result = engine.evaluate_rag_system(data['query'], data['context'], data['response'])
        evaluation_results.append(result)
    
    # Test different aggregation methods
    aggregation_methods = ['mean', 'median', 'min', 'max']
    
    print(f"\nAggregation Results:")
    for method in aggregation_methods:
        print(f"\n{method.upper()} Aggregation:")
        aggregated = engine.aggregate_evaluation_results(evaluation_results, method)
        for metric, value in aggregated.items():
            print(f"  {metric}: {value:.3f}")


def main():
    """Run all advanced evaluation capability demos."""
    print("🚀 Advanced RAG Evaluation Capabilities Demo")
    print("This demo showcases the advanced evaluation features implemented in task 8.")
    
    try:
        # Run all demos
        demo_ab_testing()
        demo_detailed_reporting()
        demo_threshold_flagging()
        demo_trend_analysis()
        demo_aggregation_methods()
        
        print("\n" + "=" * 60)
        print("✅ All advanced evaluation capabilities demonstrated successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()