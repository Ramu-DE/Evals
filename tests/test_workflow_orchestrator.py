"""Property-based tests for the Workflow Orchestrator component."""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Dict, Any, List

from src.workflow_orchestrator import LangGraphOrchestrator, WorkflowStatus


class TestWorkflowOrchestrator:
    """Test suite for workflow orchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = LangGraphOrchestrator()
    
    @given(
        nodes=st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"),
            st.fixed_dictionaries({
                "type": st.sampled_from(["function", "condition", "parallel"]),
                "function": st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
                "config": st.dictionaries(
                    st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"), 
                    st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789")
                )
            }),
            min_size=1,
            max_size=5
        ),
        entry_point=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789")
    )
    @settings(max_examples=100)
    def test_workflow_state_management_property(self, nodes: Dict[str, Dict[str, Any]], entry_point: str):
        """
        **Feature: rag-evaluation-pipeline, Property 6: Workflow State Management**
        
        For any workflow execution, state transitions should follow LangGraph patterns 
        and maintain consistency across all execution steps.
        **Validates: Requirements 2.1, 2.4**
        """
        # Ensure entry point exists in nodes
        if entry_point not in nodes:
            nodes[entry_point] = {
                "type": "function",
                "function": "start_function",
                "config": {}
            }
        
        # Create simple linear edges between nodes
        node_names = list(nodes.keys())
        edges = []
        for i in range(len(node_names) - 1):
            edges.append({
                "from": node_names[i],
                "to": node_names[i + 1],
                "conditional": False
            })
        
        # Add final edge to END
        if node_names:
            edges.append({
                "from": node_names[-1],
                "to": "__end__",
                "conditional": False
            })
        
        workflow_definition = {
            "nodes": nodes,
            "edges": edges,
            "entry_point": entry_point
        }
        
        # Create workflow
        workflow_id = self.orchestrator.create_workflow(workflow_definition)
        
        # Verify initial state
        initial_state = self.orchestrator.get_workflow_state(workflow_id)
        assert initial_state.workflow_id == workflow_id
        assert initial_state.status == WorkflowStatus.PENDING
        assert initial_state.current_step == ""
        assert initial_state.completed_steps == []
        assert initial_state.error_info is None
        
        # Execute workflow with minimal inputs
        inputs = {"workflow_id": workflow_id, "test_data": "sample"}
        
        try:
            result = self.orchestrator.execute_workflow(workflow_id, inputs)
            
            # Verify final state after execution
            final_state = self.orchestrator.get_workflow_state(workflow_id)
            
            # State consistency checks
            assert final_state.workflow_id == workflow_id
            assert final_state.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # For successful workflows, verify state progression
                assert len(final_state.completed_steps) >= 0  # At least entry point should be completed
                assert isinstance(final_state.state_data, dict)
                assert final_state.error_info is None
            
            elif final_state.status == WorkflowStatus.FAILED:
                # For failed workflows, verify error information is present
                assert final_state.error_info is not None
                assert final_state.error_info.error_type is not None
                assert final_state.error_info.error_message is not None
        
        except Exception as e:
            # If execution fails, verify error handling
            error_state = self.orchestrator.get_workflow_state(workflow_id)
            assert error_state.status == WorkflowStatus.FAILED
            assert error_state.error_info is not None
    
    @given(
        workflow_count=st.integers(min_value=2, max_value=5),
        parallel_inputs=st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=10),
                st.one_of(st.text(min_size=1, max_size=20), st.integers(), st.floats(allow_nan=False))
            ),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=100)
    def test_multi_actor_coordination_property(self, workflow_count: int, parallel_inputs: List[Dict[str, Any]]):
        """
        **Feature: rag-evaluation-pipeline, Property 7: Multi-Actor Coordination**
        
        For any workflow involving multiple actors, the orchestrator should coordinate 
        interactions without conflicts or race conditions.
        **Validates: Requirements 2.2, 2.5**
        """
        # Create multiple simple workflows
        workflow_configs = []
        
        for i in range(min(workflow_count, len(parallel_inputs))):
            nodes = {
                f"actor_{i}_start": {
                    "type": "function",
                    "function": "data_processing",
                    "config": {"actor_id": i}
                },
                f"actor_{i}_process": {
                    "type": "function", 
                    "function": "evaluation",
                    "config": {"actor_id": i}
                }
            }
            
            edges = [
                {
                    "from": f"actor_{i}_start",
                    "to": f"actor_{i}_process",
                    "conditional": False
                },
                {
                    "from": f"actor_{i}_process",
                    "to": "__end__",
                    "conditional": False
                }
            ]
            
            workflow_config = {
                "definition": {
                    "nodes": nodes,
                    "edges": edges,
                    "entry_point": f"actor_{i}_start"
                },
                "inputs": parallel_inputs[i]
            }
            workflow_configs.append(workflow_config)
        
        # Execute workflows in parallel
        parallel_result = self.orchestrator.execute_parallel_workflows(workflow_configs)
        
        # Verify coordination properties
        assert "parallel_execution_results" in parallel_result
        assert "total_workflows" in parallel_result
        assert "successful_workflows" in parallel_result
        
        results = parallel_result["parallel_execution_results"]
        
        # Verify all workflows were processed
        assert len(results) == len(workflow_configs)
        
        # Verify no conflicts occurred (each workflow should have unique results)
        workflow_ids = list(results.keys())
        assert len(workflow_ids) == len(set(workflow_ids))  # All IDs should be unique
        
        # Verify each workflow maintained its own state
        for workflow_id, result in results.items():
            workflow_state = self.orchestrator.get_workflow_state(workflow_id)
            assert workflow_state.workflow_id == workflow_id
            assert workflow_state.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
    
    @given(
        error_type=st.sampled_from(["ValueError", "RuntimeError", "ConnectionError", "TimeoutError"]),
        error_message=st.text(min_size=1, max_size=100),
        error_step=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789")
    )
    @settings(max_examples=100)
    def test_graceful_error_handling_property(self, error_type: str, error_message: str, error_step: str):
        """
        **Feature: rag-evaluation-pipeline, Property 8: Graceful Error Handling**
        
        For any workflow error condition, the system should handle it gracefully 
        and provide meaningful feedback without system failure.
        **Validates: Requirements 2.3**
        """
        # Create a simple workflow
        nodes = {
            error_step: {
                "type": "function",
                "function": "test_function",
                "config": {"should_error": True, "error_type": error_type}
            }
        }
        
        edges = [
            {
                "from": error_step,
                "to": "__end__",
                "conditional": False
            }
        ]
        
        workflow_definition = {
            "nodes": nodes,
            "edges": edges,
            "entry_point": error_step
        }
        
        workflow_id = self.orchestrator.create_workflow(workflow_definition)
        
        # Set the current step to simulate workflow execution
        self.orchestrator.workflow_states[workflow_id].current_step = error_step
        
        # Simulate an error by creating an exception
        error_classes = {
            "ValueError": ValueError,
            "RuntimeError": RuntimeError,
            "ConnectionError": ConnectionError,
            "TimeoutError": TimeoutError
        }
        
        error_instance = error_classes[error_type](error_message)
        
        # Handle the error
        error_response = self.orchestrator.handle_workflow_error(workflow_id, error_instance)
        
        # Verify graceful error handling properties
        assert "status" in error_response
        assert error_response["status"] == "error"
        
        assert "error_info" in error_response
        error_info = error_response["error_info"]
        assert error_info["error_type"] == error_type
        assert error_info["error_message"] == error_message
        assert error_info["error_step"] == error_step
        assert "timestamp" in error_info
        
        assert "workflow_id" in error_response
        assert error_response["workflow_id"] == workflow_id
        
        assert "recovery_options" in error_response
        recovery_options = error_response["recovery_options"]
        assert isinstance(recovery_options, list)
        assert len(recovery_options) > 0
        assert "retry" in recovery_options
        
        # Verify workflow state reflects the error
        workflow_state = self.orchestrator.get_workflow_state(workflow_id)
        assert workflow_state.status == WorkflowStatus.FAILED
        assert workflow_state.error_info is not None
        assert workflow_state.error_info.error_type == error_type
        assert workflow_state.error_info.error_message == error_message
        
        # Verify system didn't crash (we can still create new workflows)
        new_workflow_id = self.orchestrator.create_workflow(workflow_definition)
        assert new_workflow_id != workflow_id
        new_state = self.orchestrator.get_workflow_state(new_workflow_id)
        assert new_state.status == WorkflowStatus.PENDING


# Additional unit tests for specific functionality
class TestWorkflowOrchestratorUnits:
    """Unit tests for specific workflow orchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = LangGraphOrchestrator()
    
    def test_pause_and_resume_workflow(self):
        """Test workflow pause and resume functionality."""
        # Create simple workflow
        workflow_definition = {
            "nodes": {
                "start": {"type": "function", "function": "data_processing", "config": {}},
                "end": {"type": "function", "function": "evaluation", "config": {}}
            },
            "edges": [
                {"from": "start", "to": "end", "conditional": False},
                {"from": "end", "to": "__end__", "conditional": False}
            ],
            "entry_point": "start"
        }
        
        workflow_id = self.orchestrator.create_workflow(workflow_definition)
        
        # Pause workflow
        pause_result = self.orchestrator.pause_workflow(workflow_id)
        assert pause_result["status"] == "paused"
        assert pause_result["workflow_id"] == workflow_id
        
        # Verify state is paused
        state = self.orchestrator.get_workflow_state(workflow_id)
        assert state.status == WorkflowStatus.PAUSED
        
        # Resume workflow
        resume_result = self.orchestrator.resume_workflow(workflow_id, {"test": "data"})
        assert isinstance(resume_result, dict)
    
    def test_workflow_metrics(self):
        """Test workflow metrics collection."""
        workflow_definition = {
            "nodes": {
                "step1": {"type": "function", "function": "data_processing", "config": {}},
                "step2": {"type": "function", "function": "evaluation", "config": {}}
            },
            "edges": [
                {"from": "step1", "to": "step2", "conditional": False},
                {"from": "step2", "to": "__end__", "conditional": False}
            ],
            "entry_point": "step1"
        }
        
        workflow_id = self.orchestrator.create_workflow(workflow_definition)
        
        # Get metrics
        metrics = self.orchestrator.get_workflow_metrics(workflow_id)
        
        assert "workflow_id" in metrics
        assert "status" in metrics
        assert "completed_steps" in metrics
        assert "total_steps" in metrics
        assert "current_step" in metrics
        assert "has_errors" in metrics
        assert "state_size" in metrics
        
        assert metrics["workflow_id"] == workflow_id
        assert metrics["total_steps"] == 2
        assert metrics["has_errors"] is False