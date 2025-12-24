"""Base interfaces for the UI component."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class UserInterface(ABC):
    """Abstract base class for user interfaces."""
    
    @abstractmethod
    def display_dashboard(self) -> None:
        """Display the main dashboard."""
        pass
    
    @abstractmethod
    def render_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Render evaluation results with visualizations."""
        pass
    
    @abstractmethod
    def show_configuration_interface(self) -> Dict[str, Any]:
        """Show configuration interface and return settings."""
        pass
    
    @abstractmethod
    def display_real_time_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display real-time performance metrics."""
        pass