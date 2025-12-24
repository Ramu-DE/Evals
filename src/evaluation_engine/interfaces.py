"""Base interfaces for the Evaluation Engine component."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class EvaluationMode(Enum):
    """Evaluation modes supported by the system."""
    BINARY = "binary"
    NUMERICAL = "numerical"
    RAG_TRIAD = "rag_triad"
    LLM_JUDGE = "llm_judge"


@dataclass
class TriadMetrics:
    """Metrics for RAG-Triad evaluation."""
    retrieval_quality: float
    generation_quality: float
    overall_performance: float
    component_scores: Dict[str, float]


@dataclass
class ABTestResult:
    """Result of A/B testing comparison."""
    test_id: str
    timestamp: datetime
    system_a_metrics: Dict[str, float]
    system_b_metrics: Dict[str, float]
    comparison_results: Dict[str, str]  # 'A', 'B', or 'tie' for each metric
    statistical_significance: Dict[str, bool]
    overall_winner: str  # 'A', 'B', or 'tie'
    confidence_level: float
    metadata: Dict[str, Any]


@dataclass
class EvaluationReport:
    """Detailed evaluation report with criterion breakdowns."""
    report_id: str
    timestamp: datetime
    overall_metrics: Dict[str, float]
    criterion_breakdowns: Dict[str, Dict[str, Any]]
    performance_summary: Dict[str, Any]
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Result of an evaluation operation."""
    evaluation_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    binary_results: Dict[str, bool]
    triad_scores: Optional[TriadMetrics]
    metadata: Dict[str, Any]
    trace_id: str


class EvaluationEngine(ABC):
    """Abstract base class for evaluation engines."""
    
    @abstractmethod
    def evaluate_rag_system(self, query: str, context: List[str], response: str) -> EvaluationResult:
        """Evaluate a RAG system response."""
        pass
    
    @abstractmethod
    def run_binary_evaluation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Run binary (pass/fail) evaluation."""
        pass
    
    @abstractmethod
    def execute_rag_triad(self, dataset: List[Dict[str, Any]]) -> TriadMetrics:
        """Execute RAG-Triad evaluation."""
        pass
    
    @abstractmethod
    def llm_as_judge_evaluation(self, responses: List[str], criteria: List[str]) -> Dict[str, Any]:
        """Execute LLM-as-a-Judge evaluation."""
        pass
    
    @abstractmethod
    def run_ab_test(self, system_a_results: List[EvaluationResult], system_b_results: List[EvaluationResult]) -> ABTestResult:
        """Run A/B testing comparison between two RAG systems."""
        pass
    
    @abstractmethod
    def generate_detailed_report(self, evaluation_results: List[EvaluationResult], criteria: List[str]) -> EvaluationReport:
        """Generate detailed evaluation report with criterion breakdowns."""
        pass
    
    @abstractmethod
    def check_performance_thresholds(self, evaluation_results: List[EvaluationResult], thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Check evaluation results against performance thresholds and flag issues."""
        pass
    
    @abstractmethod
    def aggregate_evaluation_results(self, evaluation_results: List[EvaluationResult], aggregation_method: str = 'mean') -> Dict[str, float]:
        """Aggregate multiple evaluation results using specified method."""
        pass
    
    @abstractmethod
    def analyze_evaluation_trends(self, historical_results: List[List[EvaluationResult]], time_periods: List[str]) -> Dict[str, Any]:
        """Analyze trends in evaluation results over time."""
        pass