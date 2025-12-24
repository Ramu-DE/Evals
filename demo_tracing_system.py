#!/usr/bin/env python3
"""
Demo script for the RAG Evaluation Pipeline Tracing System.

This script demonstrates the key features of the OPIK-based tracing system:
- Trace creation and management
- Event logging with context
- Performance monitoring
- Alert configuration and triggering
"""

import time
from datetime import datetime
from src.tracing_system import OpikTracingSystem, AlertConfig

def main():
    print("🚀 RAG Evaluation Pipeline - Tracing System Demo")
    print("=" * 60)
    
    # Note: In a real environment, you would use actual OPIK credentials
    # For demo purposes, we'll use mock credentials
    try:
        tracer = OpikTracingSystem(
            api_key="demo_api_key",
            workspace="demo_workspace",
            project_name="rag-evaluation-demo"
        )
        print("❌ OPIK SDK not available - using mock implementation for demo")
    except ImportError:
        print("ℹ️  OPIK SDK not available - using mock implementation for demo")
        # Create a mock tracer for demonstration
        tracer = OpikTracingSystem.__new__(OpikTracingSystem)
        tracer.api_key = "demo_api_key"
        tracer.workspace = "demo_workspace"
        tracer.project_name = "rag-evaluation-demo"
        
        # Mock client for demo
        class MockClient:
            def trace(self, name, project_name, metadata):
                return MockTrace(name, project_name, metadata)
        
        class MockTrace:
            def __init__(self, name, project_name, metadata):
                self.name = name
                self.project_name = project_name
                self.metadata = metadata
                self.logs = []
            
            def log(self, **kwargs):
                self.logs.append(kwargs)
            
            def end(self, output, metadata=None):
                pass
        
        tracer.client = MockClient()
        tracer._active_traces = {}
        tracer._performance_metrics = []
        tracer._alert_configs = []
        tracer._error_history = []
        tracer._triggered_alerts = []
    
    # Demo 1: Basic Tracing
    print("\n📊 Demo 1: Basic Trace Creation and Event Logging")
    print("-" * 50)
    
    # Start a trace for a RAG evaluation
    trace_id = tracer.start_trace("rag_evaluation", {
        "model": "gpt-4",
        "dataset": "qa_pairs_v1",
        "evaluation_type": "rag_triad"
    })
    print(f"✅ Started trace: {trace_id}")
    
    # Log retrieval event
    retrieval_event = {
        "type": "retrieval",
        "message": "Retrieved relevant documents",
        "metadata": {
            "query": "What is machine learning?",
            "documents_found": 5,
            "retrieval_time_ms": 150
        }
    }
    event_id_1 = tracer.log_event(trace_id, retrieval_event)
    print(f"📝 Logged retrieval event: {event_id_1}")
    
    # Simulate some processing time
    time.sleep(0.1)
    
    # Log generation event
    generation_event = {
        "type": "generation",
        "message": "Generated response using LLM",
        "metadata": {
            "model_response_length": 250,
            "generation_time_ms": 800,
            "tokens_used": 180
        }
    }
    event_id_2 = tracer.log_event(trace_id, generation_event)
    print(f"📝 Logged generation event: {event_id_2}")
    
    # End the trace
    result = tracer.end_trace(trace_id, {
        "success": True,
        "total_time_ms": 950,
        "evaluation_score": 0.85
    })
    print(f"✅ Completed trace with duration: {result.duration_ms:.2f}ms")
    
    # Demo 2: Performance Monitoring
    print("\n📈 Demo 2: Performance Monitoring")
    print("-" * 50)
    
    # Simulate multiple operations to generate performance metrics
    for i in range(5):
        duration = 100 + (i * 50)  # Increasing latency
        has_error = i == 3  # One operation has an error
        tracer._update_performance_metrics(duration, has_error)
        print(f"📊 Operation {i+1}: {duration}ms, Error: {has_error}")
    
    # Get performance metrics
    metrics = tracer.get_performance_metrics()
    if metrics:
        latest = metrics[-1]
        print(f"📊 Latest Metrics:")
        print(f"   - Latency: {latest.latency_ms:.1f}ms")
        print(f"   - Throughput: {latest.throughput_ops_per_sec:.1f} ops/sec")
        print(f"   - Error Rate: {latest.error_rate:.1%}")
    
    # Demo 3: Alert Configuration and Triggering
    print("\n🚨 Demo 3: Alert System")
    print("-" * 50)
    
    # Configure alerts
    latency_alert = AlertConfig(
        name="high_latency_alert",
        metric_type="latency_ms",
        threshold=200.0,
        comparison="gt",
        enabled=True
    )
    tracer.add_alert_config(latency_alert)
    print(f"⚙️  Configured alert: {latency_alert.name} (latency > {latency_alert.threshold}ms)")
    
    error_rate_alert = AlertConfig(
        name="high_error_rate_alert",
        metric_type="error_rate",
        threshold=0.15,
        comparison="gt",
        enabled=True
    )
    tracer.add_alert_config(error_rate_alert)
    print(f"⚙️  Configured alert: {error_rate_alert.name} (error rate > {error_rate_alert.threshold:.1%})")
    
    # Simulate a high-latency operation that should trigger an alert
    tracer._update_performance_metrics(350.0, False)  # High latency
    tracer._check_alerts()
    
    # Check for triggered alerts
    if hasattr(tracer, '_triggered_alerts') and tracer._triggered_alerts:
        for alert in tracer._triggered_alerts:
            print(f"🚨 ALERT TRIGGERED: {alert['alert_name']}")
            print(f"   - Metric: {alert['metric_type']} = {alert['current_value']}")
            print(f"   - Threshold: {alert['threshold']}")
            print(f"   - Message: {alert['message']}")
    else:
        print("✅ No alerts triggered")
    
    # Demo 4: Query Traces
    print("\n🔍 Demo 4: Trace Querying")
    print("-" * 50)
    
    # Start another trace for querying demo
    trace_id_2 = tracer.start_trace("rag_evaluation", {
        "model": "claude-3",
        "dataset": "qa_pairs_v2"
    })
    
    # Query active traces
    active_traces = tracer.query_traces({"status": "active"})
    print(f"🔍 Found {len(active_traces)} active traces")
    
    for trace in active_traces:
        print(f"   - Trace {trace.trace_id[:8]}... (duration: {trace.duration_ms:.1f}ms)")
    
    # Clean up
    tracer.end_trace(trace_id_2, {"success": True})
    
    print("\n🎉 Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- ✅ Comprehensive trace creation and management")
    print("- ✅ Event logging with rich context and metadata")
    print("- ✅ Real-time performance monitoring")
    print("- ✅ Configurable alerting system")
    print("- ✅ Trace querying and filtering")
    print("- ✅ Integration-ready OPIK client interface")

if __name__ == "__main__":
    main()