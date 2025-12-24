"""Base interfaces for the Tracing System component."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class TraceEvent:
    """Individual event within a trace."""
    event_id: str
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class TraceResult:
    """Result of a completed trace."""
    trace_id: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    events: List[TraceEvent]
    status: str
    result_data: Dict[str, Any]


class TracingSystem(ABC):
    """Abstract base class for tracing systems."""
    
    @abstractmethod
    def start_trace(self, operation_name: str, metadata: Dict[str, Any]) -> str:
        """Start a new trace and return its ID."""
        pass
    
    @abstractmethod
    def log_event(self, trace_id: str, event_data: Dict[str, Any]) -> str:
        """Log an event within a trace."""
        pass
    
    @abstractmethod
    def end_trace(self, trace_id: str, result: Dict[str, Any]) -> TraceResult:
        """End a trace and return the result."""
        pass
    
    @abstractmethod
    def query_traces(self, filters: Dict[str, Any]) -> List[TraceResult]:
        """Query traces based on filters."""
        pass