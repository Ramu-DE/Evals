#!/usr/bin/env python3
"""Demo script for the Workflow Orchestrator component."""

from src.workflow_orchestrator import LangGraphOrchestrator, WorkflowStatus


def main():
    """Demonstrate workflow orchestrator functionality."""
    print("=== RAG Evaluation Pipeline - Workflow Orchestrator Demo ===\n")
    
    # Initialize orchestrator
    orchestrator = LangGraphOrchestrator()
    print("✓ Workflow orchestrator initialized")
    
    # Create a simple evaluation workflow
    workflow_definition = {
        "nodes": {
            "data_ingestion": {
                "type": "function",
                "function": "data_processing",
                "config": {"source": "documents"}
            },
            "vector_embedding": {
                "type": "function", 
                "function": "evaluation",
                "config": {"model": "sentence-transformers"}
            },
            "evaluation": {
                "type": "function",
                "function": "aggregation", 
                "config": {"metrics": ["precision", "recall"]}
            }
        },
        "edges": [
            {"from": "data_ingestion", "to": "vector_embedding", "conditional": False},
            {"from": "vector_embedding", "to": "evaluation", "conditional": False},
            {"from": "evaluation", "to": "__end__", "conditional": False}
        ],
        "entry_point": "data_ingestion"
    }
    
    # Create workflow
    workflow_id = orchestrator.create_workflow(workflow_definition)
    print(f"✓ Created workflow: {workflow_id}")
    
    # Check initial state
    initial_state = orchestrator.get_workflow_state(workflow_id)
    print(f"✓ Initial workflow status: {initial_state.status.value}")
    
    # Execute workflow
    inputs = {
        "workflow_id": workflow_id,
        "documents": ["doc1.txt", "doc2.txt"],
        "evaluation_criteria": ["relevance", "accuracy"]
    }
    
    print("\n--- Executing Workflow ---")
    result = orchestrator.execute_workflow(workflow_id, inputs)
    print(f"✓ Workflow execution result: {result}")
    
    # Check final state
    final_state = orchestrator.get_workflow_state(workflow_id)
    print(f"✓ Final workflow status: {final_state.status.value}")
    print(f"✓ Completed steps: {final_state.completed_steps}")
    
    # Get workflow metrics
    metrics = orchestrator.get_workflow_metrics(workflow_id)
    print(f"\n--- Workflow Metrics ---")
    print(f"Total steps: {metrics['total_steps']}")
    print(f"Completed steps: {metrics['completed_steps']}")
    print(f"Current step: {metrics['current_step']}")
    print(f"Has errors: {metrics['has_errors']}")
    
    # Demonstrate error handling
    print(f"\n--- Error Handling Demo ---")
    error = ValueError("Simulated evaluation error")
    error_response = orchestrator.handle_workflow_error(workflow_id, error)
    print(f"✓ Error handled gracefully: {error_response['status']}")
    print(f"✓ Recovery options: {error_response['recovery_options']}")
    
    # Demonstrate parallel execution
    print(f"\n--- Parallel Execution Demo ---")
    parallel_configs = [
        {
            "definition": {
                "nodes": {"task1": {"type": "function", "function": "data_processing", "config": {}}},
                "edges": [{"from": "task1", "to": "__end__", "conditional": False}],
                "entry_point": "task1"
            },
            "inputs": {"data": "dataset_1"}
        },
        {
            "definition": {
                "nodes": {"task2": {"type": "function", "function": "evaluation", "config": {}}},
                "edges": [{"from": "task2", "to": "__end__", "conditional": False}],
                "entry_point": "task2"
            },
            "inputs": {"data": "dataset_2"}
        }
    ]
    
    parallel_result = orchestrator.execute_parallel_workflows(parallel_configs)
    print(f"✓ Parallel execution completed")
    print(f"✓ Total workflows: {parallel_result['total_workflows']}")
    print(f"✓ Successful workflows: {parallel_result['successful_workflows']}")
    
    print(f"\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()