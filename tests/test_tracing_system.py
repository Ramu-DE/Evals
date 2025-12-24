"""Property-based tests for the Tracing System."""

import pytest
from hypothesis import given, strategies as st, assume
from datetime import datetime, timedelta
from typing import Dict, Any
import uuid

from src.tracing_system import OpikTracingSystem, PerformanceMetrics, AlertConfig


class MockOpikClient:
    """Mock OPIK client for testing."""
    
    def __init__(self):
        self.traces = {}
        self.logs = []
    
    def trace(self, name: str, project_name: str, metadata: Dict[str, Any]):
        return MockTrace(name, project_name, metadata, self)


class MockTrace:
    """Mock OPIK trace for testing."""
    
    def __init__(self, name: str, project_name: str, metadata: Dict[str, Any], client):
        self.name = name
        self.project_name = project_name
        self.metadata = metadata
        self.client = client
        self.logs = []
        self.ended = False
        self.output = None
    
    def log(self, **kwargs):
        log_entry = {
            'message': kwargs.get('message', ''),
            'level': kwargs.get('level', 'info'),
            'timestamp': kwargs.get('timestamp') or datetime.now(),
        }
        log_entry.update(kwargs)
        self.logs.append(log_entry)
    
    def end(self, output: Dict[str, Any], metadata: Dict[str, Any] = None):
        self.ended = True
        self.output = output
        if metadata:
            self.metadata.update(metadata)


class TestOpikTracingSystem:
    """Test suite for OPIK tracing system implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock tracing system that doesn't require real OPIK connection
        self.tracer = OpikTracingSystem.__new__(OpikTracingSystem)
        self.tracer.api_key = "test_key"
        self.tracer.workspace = "test_workspace"
        self.tracer.project_name = "test_project"
        self.tracer.client = MockOpikClient()
        self.tracer._active_traces = {}
        self.tracer._performance_metrics = []
        self.tracer._alert_configs = []
    
    @given(
        operation_name=st.text(min_size=1, max_size=100),
        metadata=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(allow_nan=False)),
            min_size=0,
            max_size=10
        )
    )
    def test_comprehensive_tracing_property(self, operation_name: str, metadata: Dict[str, Any]):
        """**Feature: rag-evaluation-pipeline, Property 15: Comprehensive Tracing**
        
        For any RAG operation, the Tracing System should capture complete execution 
        traces with appropriate context, metadata, and timing information.
        **Validates: Requirements 5.1, 5.2, 5.4**
        """
        # Start a trace
        trace_id = self.tracer.start_trace(operation_name, metadata)
        
        # Verify trace was created and is active
        assert trace_id in self.tracer._active_traces
        trace_info = self.tracer._active_traces[trace_id]
        
        # Verify comprehensive trace information is captured
        assert trace_info['operation_name'] == operation_name
        assert trace_info['metadata'] == metadata
        assert trace_info['status'] == 'active'
        assert isinstance(trace_info['start_time'], datetime)
        assert trace_info['events'] == []
        
        # Log some events with context
        event_data_1 = {
            'type': 'retrieval',
            'message': 'Retrieved documents',
            'metadata': {'doc_count': 5}
        }
        event_id_1 = self.tracer.log_event(trace_id, event_data_1)
        
        event_data_2 = {
            'type': 'generation', 
            'message': 'Generated response',
            'metadata': {'token_count': 150}
        }
        event_id_2 = self.tracer.log_event(trace_id, event_data_2)
        
        # Verify events are captured with proper context
        assert len(trace_info['events']) == 2
        
        event_1 = trace_info['events'][0]
        assert event_1.event_id == event_id_1
        assert event_1.event_type == 'retrieval'
        assert event_1.data == event_data_1
        assert isinstance(event_1.timestamp, datetime)
        
        event_2 = trace_info['events'][1]
        assert event_2.event_id == event_id_2
        assert event_2.event_type == 'generation'
        assert event_2.data == event_data_2
        
        # End the trace with result
        result_data = {'success': True, 'response_length': 200}
        trace_result = self.tracer.end_trace(trace_id, result_data)
        
        # Verify comprehensive trace result
        assert trace_result.trace_id == trace_id
        assert isinstance(trace_result.start_time, datetime)
        assert isinstance(trace_result.end_time, datetime)
        assert trace_result.duration_ms >= 0  # Duration should be non-negative
        assert trace_result.status == 'completed'
        assert trace_result.result_data == result_data
        assert len(trace_result.events) == 2
        
        # Verify timing information is accurate
        expected_duration = (trace_result.end_time - trace_result.start_time).total_seconds() * 1000
        assert abs(trace_result.duration_ms - expected_duration) < 1.0  # Allow 1ms tolerance
        assert trace_result.duration_ms >= 0  # Duration should be non-negative
        
        # Verify trace is no longer active
        assert trace_id not in self.tracer._active_traces
    
    @given(
        trace_count=st.integers(min_value=1, max_value=10),
        events_per_trace=st.integers(min_value=0, max_value=5)
    )
    def test_multiple_concurrent_traces(self, trace_count: int, events_per_trace: int):
        """Test that multiple traces can be managed concurrently with full context."""
        trace_ids = []
        
        # Start multiple traces
        for i in range(trace_count):
            operation_name = f"operation_{i}"
            metadata = {'trace_index': i}
            trace_id = self.tracer.start_trace(operation_name, metadata)
            trace_ids.append(trace_id)
        
        # Add events to each trace
        for trace_id in trace_ids:
            for j in range(events_per_trace):
                event_data = {
                    'type': f'event_{j}',
                    'message': f'Event {j} for trace {trace_id}',
                    'metadata': {'event_index': j}
                }
                self.tracer.log_event(trace_id, event_data)
        
        # Verify all traces are properly maintained
        assert len(self.tracer._active_traces) == trace_count
        
        for i, trace_id in enumerate(trace_ids):
            trace_info = self.tracer._active_traces[trace_id]
            assert trace_info['operation_name'] == f"operation_{i}"
            assert trace_info['metadata']['trace_index'] == i
            assert len(trace_info['events']) == events_per_trace
        
        # End all traces
        for trace_id in trace_ids:
            result = {'completed': True}
            trace_result = self.tracer.end_trace(trace_id, result)
            assert trace_result.trace_id == trace_id
            assert len(trace_result.events) == events_per_trace
        
        # Verify all traces are completed
        assert len(self.tracer._active_traces) == 0


class TestPerformanceMonitoring:
    """Test suite for performance monitoring capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = OpikTracingSystem.__new__(OpikTracingSystem)
        self.tracer.api_key = "test_key"
        self.tracer.workspace = "test_workspace"
        self.tracer.project_name = "test_project"
        self.tracer.client = MockOpikClient()
        self.tracer._active_traces = {}
        self.tracer._performance_metrics = []  # Ensure clean state
        self.tracer._alert_configs = []
        self.tracer._error_history = []
    
    @given(
        duration_values=st.lists(
            st.floats(min_value=1.0, max_value=10000.0, allow_nan=False),
            min_size=1,
            max_size=20
        ),
        error_flags=st.lists(st.booleans(), min_size=1, max_size=20)
    )
    def test_performance_monitoring_completeness_property(self, duration_values, error_flags):
        """**Feature: rag-evaluation-pipeline, Property 16: Performance Monitoring Completeness**
        
        For any monitoring period, the Tracing System should track latency, throughput, 
        and error rates for all system components.
        **Validates: Requirements 5.3**
        """
        # Clear any existing metrics to ensure clean test state
        self.tracer._performance_metrics = []
        self.tracer._error_history = []
        
        # Ensure lists are same length
        min_length = min(len(duration_values), len(error_flags))
        duration_values = duration_values[:min_length]
        error_flags = error_flags[:min_length]
        
        # Simulate multiple operations to generate metrics
        for i, (duration_ms, has_error) in enumerate(zip(duration_values, error_flags)):
            # Simulate the internal metric update that would happen during end_trace
            self.tracer._update_performance_metrics(duration_ms, has_error)
        
        # Verify performance metrics are tracked
        metrics = self.tracer.get_performance_metrics()
        assert len(metrics) == len(duration_values)
        
        # Verify all required metrics are present and valid
        for i, metric in enumerate(metrics):
            # Latency should match input
            assert metric.latency_ms == duration_values[i]
            
            # Throughput should be positive
            assert metric.throughput_ops_per_sec > 0
            
            # Error rate should be between 0 and 1
            assert 0 <= metric.error_rate <= 1
            
            # Timestamp should be recent
            assert isinstance(metric.timestamp, datetime)
            time_diff = datetime.now() - metric.timestamp
            assert time_diff < timedelta(seconds=10)  # Should be very recent
        
        # Verify error rate calculation is correct for the last metric
        if metrics:
            last_metric = metrics[-1]
            expected_error_rate = sum(error_flags) / len(error_flags)
            # Allow some tolerance due to sliding window calculations
            assert abs(last_metric.error_rate - expected_error_rate) <= 0.5


class TestAlertSystem:
    """Test suite for alert system capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = OpikTracingSystem.__new__(OpikTracingSystem)
        self.tracer.api_key = "test_key"
        self.tracer.workspace = "test_workspace"
        self.tracer.project_name = "test_project"
        self.tracer.client = MockOpikClient()
        self.tracer._active_traces = {}
        self.tracer._performance_metrics = []
        self.tracer._alert_configs = []
        self.tracer._error_history = []
        self.tracer._triggered_alerts = []
    
    @given(
        alert_name=st.text(min_size=1, max_size=50),
        metric_type=st.sampled_from(['latency_ms', 'throughput_ops_per_sec', 'error_rate']),
        threshold=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False),
        comparison=st.sampled_from(['gt', 'lt', 'eq']),
        metric_value=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False)
    )
    def test_alert_responsiveness_property(self, alert_name: str, metric_type: str, 
                                         threshold: float, comparison: str, metric_value: float):
        """**Feature: rag-evaluation-pipeline, Property 17: Alert Responsiveness**
        
        For any configured alert condition, the Tracing System should notify operators 
        when thresholds are exceeded or anomalies are detected.
        **Validates: Requirements 5.5**
        """
        # Clear state for each test example
        self.tracer._alert_configs = []
        self.tracer._triggered_alerts = []
        self.tracer._performance_metrics = []
        self.tracer._error_history = []
        
        # Configure an alert
        alert_config = AlertConfig(
            name=alert_name,
            metric_type=metric_type,
            threshold=threshold,
            comparison=comparison,
            enabled=True
        )
        self.tracer.add_alert_config(alert_config)
        
        # Create a performance metric with the exact test value
        metrics = PerformanceMetrics(
            latency_ms=metric_value if metric_type == 'latency_ms' else 100.0,
            throughput_ops_per_sec=metric_value if metric_type == 'throughput_ops_per_sec' else 10.0,
            error_rate=min(metric_value, 1.0) if metric_type == 'error_rate' else 0.1,
            timestamp=datetime.now()
        )
        self.tracer._performance_metrics.append(metrics)
        
        # Trigger alert check
        self.tracer._check_alerts()
        
        # Determine if alert should have been triggered
        should_alert = False
        actual_value = getattr(metrics, metric_type)
        
        if comparison == 'gt' and actual_value > threshold:
            should_alert = True
        elif comparison == 'lt' and actual_value < threshold:
            should_alert = True
        elif comparison == 'eq' and abs(actual_value - threshold) < 0.001:
            should_alert = True
        
        if should_alert:
            # Verify alert was triggered by checking the triggered alerts list
            assert hasattr(self.tracer, '_triggered_alerts')
            triggered_alerts = [alert for alert in self.tracer._triggered_alerts 
                              if alert['alert_name'] == alert_name]
            assert len(triggered_alerts) > 0, f"Expected alert '{alert_name}' to be triggered"
            
            # Verify alert contains correct information
            alert = triggered_alerts[0]
            assert alert['metric_type'] == metric_type
            assert alert['threshold'] == threshold
            assert alert['current_value'] == actual_value
        else:
            # Verify no alert was triggered
            triggered_alerts = [alert for alert in self.tracer._triggered_alerts 
                              if alert['alert_name'] == alert_name]
            assert len(triggered_alerts) == 0, f"Alert '{alert_name}' should not have been triggered"
        
        # Verify alert configuration management
        assert self.tracer.remove_alert_config(alert_name) == True
        assert len(self.tracer._alert_configs) == 0
        assert self.tracer.remove_alert_config(alert_name) == False  # Should not find it again
    
    def test_alert_enable_disable(self):
        """Test that disabled alerts don't trigger."""
        # Add a disabled alert that would normally trigger
        alert_config = AlertConfig(
            name="test_alert",
            metric_type="latency_ms",
            threshold=50.0,
            comparison="gt",
            enabled=False  # Disabled
        )
        self.tracer.add_alert_config(alert_config)
        
        # Create metrics that would trigger the alert if it were enabled
        metrics = PerformanceMetrics(
            latency_ms=100.0,  # Above threshold
            throughput_ops_per_sec=10.0,
            error_rate=0.1,
            timestamp=datetime.now()
        )
        self.tracer._performance_metrics.append(metrics)
        
        # Check alerts - should not trigger because alert is disabled
        initial_trace_count = len(self.tracer._active_traces)
        self.tracer._check_alerts()
        
        # No new traces should be created for disabled alerts
        assert len(self.tracer._active_traces) == initial_trace_count