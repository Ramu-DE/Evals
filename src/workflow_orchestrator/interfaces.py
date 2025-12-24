"""Base interfaces for the Workflow Orchestrator component."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ErrorInfo:
    """Information about workflow errors."""
    error_type: str
    error_message: str
    error_step: str
    timestamp: str


@dataclass
class WorkflowState:
    """State of a workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    current_step: str
    completed_steps: List[str]
    state_data: Dict[str, Any]
    error_info: Optional[ErrorInfo]


class WorkflowOrchestrator(ABC):
    """Abstract base class for workflow orchestrators."""
    
    @abstractmethod
    def create_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a new workflow and return its ID."""
        pass
    
    @abstractmethod
    def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow with given inputs."""
        pass
    
    @abstractmethod
    def get_workflow_state(self, workflow_id: str) -> WorkflowState:
        """Get the current state of a workflow."""
        pass
    
    @abstractmethod
    def handle_workflow_error(self, workflow_id: str, error: Exception) -> Dict[str, Any]:
        """Handle workflow errors gracefully."""
        pass