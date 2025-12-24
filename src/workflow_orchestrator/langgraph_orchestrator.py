"""LangGraph-based implementation of the Workflow Orchestrator."""

import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import asdict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .interfaces import WorkflowOrchestrator, WorkflowState, WorkflowStatus, ErrorInfo


class LangGraphOrchestrator(WorkflowOrchestrator):
    """LangGraph-based workflow orchestrator with state management and error handling."""
    
    def __init__(self):
        """Initialize the orchestrator with memory-based checkpointing."""
        self.checkpointer = MemorySaver()
        self.workflows: Dict[str, StateGraph] = {}
        self.workflow_states: Dict[str, WorkflowState] = {}
        self.workflow_configs: Dict[str, Dict[str, Any]] = {}
        
    def create_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a new workflow from definition and return its ID."""
        workflow_id = str(uuid.uuid4())
        
        # Extract workflow components
        nodes = workflow_definition.get("nodes", {})
        edges = workflow_definition.get("edges", [])
        entry_point = workflow_definition.get("entry_point", "start")
        
        # Create state graph
        workflow = StateGraph(dict)
        
        # Add nodes to the graph
        for node_name, node_config in nodes.items():
            node_func = self._create_node_function(node_name, node_config)
            workflow.add_node(node_name, node_func)
        
        # Add edges
        for edge in edges:
            if edge.get("conditional"):
                # Conditional edge
                workflow.add_conditional_edges(
                    edge["from"],
                    self._create_conditional_function(edge["condition"]),
                    edge["mapping"]
                )
            else:
                # Regular edge
                workflow.add_edge(edge["from"], edge["to"])
        
        # Set entry point
        workflow.set_entry_point(entry_point)
        
        # Compile the workflow
        compiled_workflow = workflow.compile(checkpointer=self.checkpointer)
        
        # Store workflow and initialize state
        self.workflows[workflow_id] = compiled_workflow
        self.workflow_configs[workflow_id] = workflow_definition
        self.workflow_states[workflow_id] = WorkflowState(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            current_step="",
            completed_steps=[],
            state_data={},
            error_info=None
        )
        
        return workflow_id
    
    def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow with given inputs."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        # Update workflow state to running
        self.workflow_states[workflow_id].status = WorkflowStatus.RUNNING
        
        try:
            # Create thread configuration for checkpointing
            thread_config = {"configurable": {"thread_id": workflow_id}}
            
            # Execute workflow
            result = workflow.invoke(inputs, config=thread_config)
            
            # Update state to completed
            self.workflow_states[workflow_id].status = WorkflowStatus.COMPLETED
            self.workflow_states[workflow_id].state_data = result
            
            return result
            
        except Exception as e:
            # Handle error
            error_response = self.handle_workflow_error(workflow_id, e)
            return error_response
    
    def get_workflow_state(self, workflow_id: str) -> WorkflowState:
        """Get the current state of a workflow."""
        if workflow_id not in self.workflow_states:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        return self.workflow_states[workflow_id]
    
    def handle_workflow_error(self, workflow_id: str, error: Exception) -> Dict[str, Any]:
        """Handle workflow errors gracefully."""
        current_step = self.workflow_states[workflow_id].current_step if workflow_id in self.workflow_states else "unknown"
        
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            error_message=str(error),
            error_step=current_step,
            timestamp=datetime.now().isoformat()
        )
        
        # Update workflow state
        if workflow_id in self.workflow_states:
            self.workflow_states[workflow_id].status = WorkflowStatus.FAILED
            self.workflow_states[workflow_id].error_info = error_info
        
        return {
            "status": "error",
            "error_info": asdict(error_info),
            "workflow_id": workflow_id,
            "recovery_options": self._get_recovery_options(workflow_id, error)
        }
    
    def pause_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Pause a running workflow."""
        if workflow_id not in self.workflow_states:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        self.workflow_states[workflow_id].status = WorkflowStatus.PAUSED
        
        return {
            "status": "paused",
            "workflow_id": workflow_id,
            "checkpoint_available": True
        }
    
    def resume_workflow(self, workflow_id: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resume a paused workflow from checkpoint."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_state = self.workflow_states[workflow_id]
        if workflow_state.status != WorkflowStatus.PAUSED:
            raise ValueError(f"Workflow {workflow_id} is not paused")
        
        # Resume from checkpoint
        workflow_state.status = WorkflowStatus.RUNNING
        
        # Continue execution with existing state or new inputs
        resume_inputs = inputs or workflow_state.state_data
        return self.execute_workflow(workflow_id, resume_inputs)
    
    def execute_parallel_workflows(self, workflow_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple workflows in parallel."""
        workflow_ids = []
        
        # Create all workflows
        for config in workflow_configs:
            workflow_id = self.create_workflow(config["definition"])
            workflow_ids.append((workflow_id, config.get("inputs", {})))
        
        # Execute workflows in parallel using asyncio
        async def run_parallel():
            tasks = []
            for workflow_id, inputs in workflow_ids:
                task = asyncio.create_task(
                    self._async_execute_workflow(workflow_id, inputs)
                )
                tasks.append((workflow_id, task))
            
            results = {}
            for workflow_id, task in tasks:
                try:
                    result = await task
                    results[workflow_id] = result
                except Exception as e:
                    results[workflow_id] = self.handle_workflow_error(workflow_id, e)
            
            return results
        
        # Run the parallel execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_parallel())
        finally:
            loop.close()
        
        return {
            "parallel_execution_results": results,
            "total_workflows": len(workflow_ids),
            "successful_workflows": sum(1 for r in results.values() if r.get("status") != "error")
        }
    
    async def _async_execute_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for workflow execution."""
        return self.execute_workflow(workflow_id, inputs)
    
    def _create_node_function(self, node_name: str, node_config: Dict[str, Any]) -> Callable:
        """Create a node function from configuration."""
        def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            # Update current step
            if node_name in self.workflow_states:
                workflow_id = state.get("workflow_id")
                if workflow_id and workflow_id in self.workflow_states:
                    self.workflow_states[workflow_id].current_step = node_name
                    if node_name not in self.workflow_states[workflow_id].completed_steps:
                        self.workflow_states[workflow_id].completed_steps.append(node_name)
            
            # Execute node logic based on type
            node_type = node_config.get("type", "function")
            
            if node_type == "function":
                # Execute custom function
                func_name = node_config.get("function")
                if func_name:
                    # This would be where custom functions are called
                    # For now, we'll simulate processing
                    result = self._execute_node_function(func_name, state, node_config)
                    state.update(result)
            
            elif node_type == "condition":
                # Handle conditional logic
                condition = node_config.get("condition")
                if condition:
                    state["condition_result"] = self._evaluate_condition(condition, state)
            
            elif node_type == "parallel":
                # Handle parallel execution within a node
                parallel_tasks = node_config.get("tasks", [])
                results = self._execute_parallel_tasks(parallel_tasks, state)
                state["parallel_results"] = results
            
            return state
        
        return node_function
    
    def _create_conditional_function(self, condition_config: Dict[str, Any]) -> Callable:
        """Create a conditional function for routing."""
        def conditional_function(state: Dict[str, Any]) -> str:
            condition_type = condition_config.get("type", "simple")
            
            if condition_type == "simple":
                field = condition_config.get("field")
                value = condition_config.get("value")
                operator = condition_config.get("operator", "equals")
                
                state_value = state.get(field)
                
                if operator == "equals":
                    return "true" if state_value == value else "false"
                elif operator == "greater_than":
                    return "true" if state_value > value else "false"
                elif operator == "less_than":
                    return "true" if state_value < value else "false"
            
            return "false"
        
        return conditional_function
    
    def _execute_node_function(self, func_name: str, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a named function within a node."""
        # This is where custom business logic would be implemented
        # For now, we'll simulate different types of operations
        
        if func_name == "data_processing":
            return {"processed_data": f"Processed: {state.get('input_data', 'no_data')}"}
        
        elif func_name == "evaluation":
            return {"evaluation_result": {"score": 0.85, "passed": True}}
        
        elif func_name == "aggregation":
            results = state.get("results", [])
            return {"aggregated_result": {"count": len(results), "average": sum(results) / len(results) if results else 0}}
        
        else:
            # Default processing
            return {"node_output": f"Executed {func_name} with config {config}"}
    
    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """Evaluate a condition string against state."""
        # Simple condition evaluation - in production this would be more sophisticated
        try:
            # Replace state variables in condition
            for key, value in state.items():
                condition = condition.replace(f"${key}", str(value))
            
            # Evaluate the condition (be careful with eval in production!)
            return bool(eval(condition))
        except:
            return False
    
    def _execute_parallel_tasks(self, tasks: List[Dict[str, Any]], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel within a node."""
        results = []
        
        for task in tasks:
            task_result = self._execute_node_function(
                task.get("function", "default"),
                state,
                task.get("config", {})
            )
            results.append(task_result)
        
        return results
    
    def _get_recovery_options(self, workflow_id: str, error: Exception) -> List[str]:
        """Get available recovery options for a failed workflow."""
        options = ["retry", "skip_step", "manual_intervention"]
        
        # Add specific recovery options based on error type
        if isinstance(error, TimeoutError):
            options.append("increase_timeout")
        
        if isinstance(error, ConnectionError):
            options.append("retry_with_backoff")
        
        return options
    
    def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get performance metrics for a workflow."""
        if workflow_id not in self.workflow_states:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        state = self.workflow_states[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "status": state.status.value,
            "completed_steps": len(state.completed_steps),
            "total_steps": len(self.workflow_configs[workflow_id].get("nodes", {})),
            "current_step": state.current_step,
            "has_errors": state.error_info is not None,
            "state_size": len(json.dumps(state.state_data))
        }