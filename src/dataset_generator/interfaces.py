"""Base interfaces for the Dataset Generator component."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class EvaluationDataset:
    """Evaluation dataset structure."""
    dataset_id: str
    questions: List[str]
    contexts: List[List[str]]
    ground_truth_answers: List[str]
    metadata: Dict[str, Any]
    quality_score: float


@dataclass
class QualityReport:
    """Quality assessment report for datasets."""
    overall_score: float
    quality_metrics: Dict[str, float]
    issues_found: List[str]
    recommendations: List[str]


class DatasetGenerator(ABC):
    """Abstract base class for dataset generators."""
    
    @abstractmethod
    def generate_from_knowledge_graph(self, graph_data: Dict[str, Any]) -> EvaluationDataset:
        """Generate evaluation dataset from knowledge graph."""
        pass
    
    @abstractmethod
    def create_synthetic_dataset(self, domain: str, size: int) -> EvaluationDataset:
        """Create synthetic evaluation dataset."""
        pass
    
    @abstractmethod
    def validate_dataset_quality(self, dataset: EvaluationDataset) -> QualityReport:
        """Validate the quality of a dataset."""
        pass
    
    @abstractmethod
    def export_dataset(self, dataset: EvaluationDataset, format: str) -> Dict[str, Any]:
        """Export dataset in specified format."""
        pass