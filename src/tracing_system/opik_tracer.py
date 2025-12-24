"""OPIK-based implementation of the TracingSystem interface."""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .interfaces import TracingSystem, TraceEvent, TraceResult

try:
    import opik
    from opik import Opik
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlertConfig:
    """Configuration for alerting."""
    name: str
    metric_type: str  # 'latency', 'throughput', 'error_rate'
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    enabled: bool = True


class OpikTracingSystem(TracingSystem):
    """OPIK-based implementation of the TracingSystem interface."""
    
    def __init__(self, api_key: str, workspace: str, project_name: str = "rag-evaluation"):
        """Initialize OPIK tracing system.
        
        Args:
            api_key: OPIK API key for authentication
            workspace: OPIK workspace identifier
            project_name: Name of the project for organizing traces
        """
        if not OPIK_AVAILABLE:
            raise ImportError("OPIK SDK is not available. Please install it with: pip install opik")
        
        self.api_key = api_key
        self.workspace = workspace
        self.project_name = project_name
        
        # Initialize OPIK client
        self.client = Opik(api_key=api_key, workspace=workspace)
        
        # Internal state for tracking active traces
        self._active_traces: Dict[str, Dict[str, Any]] = {}
        self._performance_metrics: List[PerformanceMetrics] = []
        self._alert_configs: List[AlertConfig] = []
        self._error_history: List[bool] = []
        self._triggered_alerts: List[Dict[str, Any]] = []
        
    def start_trace(self, operation_name: str, metadata: Dict[str, Any]) -> str:
        """Start a new trace and return its ID.
        
        Args:
            operation_name: Name of the operation being traced
            metadata: Additional metadata for the trace
            
        Returns:
            Unique trace identifier
        """
        trace_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Create OPIK trace
        trace = self.client.trace(
            name=operation_name,
            project_name=self.project_name,
            metadata=metadata
        )
        
        # Store trace information locally
        self._active_traces[trace_id] = {
            'opik_trace': trace,
            'operation_name': operation_name,
            'start_time': start_time,
            'metadata': metadata,
            'events': [],
            'status': 'active'
        }
        
        return trace_id
    
    def log_event(self, trace_id: str, event_data: Dict[str, Any]) -> str:
        """Log an event within a trace.
        
        Args:
            trace_id: ID of the trace to log the event to
            event_data: Data describing the event
            
        Returns:
            Event identifier
        """
        if trace_id not in self._active_traces:
            raise ValueError(f"Trace {trace_id} not found or already completed")
        
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create event object
        event = TraceEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_data.get('type', 'generic'),
            data=event_data,
            metadata=event_data.get('metadata', {})
        )
        
        # Add to local trace
        self._active_traces[trace_id]['events'].append(event)
        
        # Log to OPIK
        opik_trace = self._active_traces[trace_id]['opik_trace']
        log_data = {
            'message': event_data.get('message', ''),
            'level': event_data.get('level', 'info'),
            'timestamp': timestamp,
        }
        log_data.update({k: v for k, v in event_data.items() if k not in ['message', 'level']})
        opik_trace.log(**log_data)
        
        return event_id
    
    def end_trace(self, trace_id: str, result: Dict[str, Any]) -> TraceResult:
        """End a trace and return the result.
        
        Args:
            trace_id: ID of the trace to end
            result: Final result data for the trace
            
        Returns:
            Complete trace result
        """
        if trace_id not in self._active_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        trace_info = self._active_traces[trace_id]
        end_time = datetime.now()
        duration_ms = (end_time - trace_info['start_time']).total_seconds() * 1000
        
        # Update OPIK trace with final result
        opik_trace = trace_info['opik_trace']
        opik_trace.end(
            output=result,
            metadata={'duration_ms': duration_ms}
        )
        
        # Create trace result
        trace_result = TraceResult(
            trace_id=trace_id,
            start_time=trace_info['start_time'],
            end_time=end_time,
            duration_ms=duration_ms,
            events=trace_info['events'],
            status='completed',
            result_data=result
        )
        
        # Update performance metrics
        self._update_performance_metrics(duration_ms, result.get('error', False))
        
        # Check alerts
        self._check_alerts()
        
        # Remove from active traces
        del self._active_traces[trace_id]
        
        return trace_result
    
    def query_traces(self, filters: Dict[str, Any]) -> List[TraceResult]:
        """Query traces based on filters.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of matching trace results
        """
        # This is a simplified implementation
        # In a real system, this would query the OPIK backend
        results = []
        
        # For now, return active traces that match filters
        for trace_id, trace_info in self._active_traces.items():
            if self._matches_filters(trace_info, filters):
                # Create a partial result for active traces
                current_time = datetime.now()
                duration_ms = (current_time - trace_info['start_time']).total_seconds() * 1000
                
                result = TraceResult(
                    trace_id=trace_id,
                    start_time=trace_info['start_time'],
                    end_time=current_time,
                    duration_ms=duration_ms,
                    events=trace_info['events'],
                    status=trace_info['status'],
                    result_data={}
                )
                results.append(result)
        
        return results
    
    def get_performance_metrics(self) -> List[PerformanceMetrics]:
        """Get current performance metrics.
        
        Returns:
            List of performance metrics
        """
        return self._performance_metrics.copy()
    
    def add_alert_config(self, config: AlertConfig) -> None:
        """Add an alert configuration.
        
        Args:
            config: Alert configuration to add
        """
        self._alert_configs.append(config)
    
    def remove_alert_config(self, alert_name: str) -> bool:
        """Remove an alert configuration.
        
        Args:
            alert_name: Name of the alert to remove
            
        Returns:
            True if alert was removed, False if not found
        """
        for i, config in enumerate(self._alert_configs):
            if config.name == alert_name:
                del self._alert_configs[i]
                return True
        return False
    
    def _update_performance_metrics(self, duration_ms: float, has_error: bool) -> None:
        """Update performance metrics with new trace data.
        
        Args:
            duration_ms: Duration of the completed trace
            has_error: Whether the trace had an error
        """
        current_time = datetime.now()
        
        # Calculate metrics over the last minute
        recent_metrics = [
            m for m in self._performance_metrics
            if (current_time - m.timestamp).total_seconds() < 60
        ]
        
        # Calculate throughput (operations per second)
        if recent_metrics:
            time_span = (current_time - recent_metrics[0].timestamp).total_seconds()
            throughput = len(recent_metrics) + 1  # Include current operation
        else:
            throughput = 1.0
        
        # Calculate error rate based on actual errors, not previous error rates
        # We need to track errors differently - for now, use a simple approach
        # In a real implementation, we'd maintain a separate error tracking mechanism
        if not hasattr(self, '_error_history'):
            self._error_history = []
        
        # Add current error to history
        self._error_history.append(has_error)
        
        # Keep only recent errors (last 100 operations for sliding window)
        self._error_history = self._error_history[-100:]
        
        # Calculate error rate from actual error history
        error_rate = sum(self._error_history) / len(self._error_history)
        
        # Add new metrics
        metrics = PerformanceMetrics(
            latency_ms=duration_ms,
            throughput_ops_per_sec=throughput,
            error_rate=error_rate,
            timestamp=current_time
        )
        
        self._performance_metrics.append(metrics)
        
        # Keep only recent metrics (last 5 minutes)
        cutoff_time = current_time.timestamp() - 300  # 5 minutes
        self._performance_metrics = [
            m for m in self._performance_metrics
            if m.timestamp.timestamp() > cutoff_time
        ]
    
    def _check_alerts(self) -> None:
        """Check if any alert conditions are met."""
        if not self._performance_metrics:
            return
        
        latest_metrics = self._performance_metrics[-1]
        
        for alert_config in self._alert_configs:
            if not alert_config.enabled:
                continue
            
            metric_value = getattr(latest_metrics, alert_config.metric_type, None)
            if metric_value is None:
                continue
            
            should_alert = False
            if alert_config.comparison == 'gt' and metric_value > alert_config.threshold:
                should_alert = True
            elif alert_config.comparison == 'lt' and metric_value < alert_config.threshold:
                should_alert = True
            elif alert_config.comparison == 'eq' and abs(metric_value - alert_config.threshold) < 0.001:
                should_alert = True
            
            if should_alert:
                self._trigger_alert(alert_config, metric_value)
    
    def _trigger_alert(self, alert_config: AlertConfig, metric_value: float) -> None:
        """Trigger an alert.
        
        Args:
            alert_config: Configuration of the triggered alert
            metric_value: Current value of the metric
        """
        # In a real implementation, this would send notifications
        # For now, we'll just log the alert without creating a trace to avoid recursion
        alert_data = {
            'type': 'alert',
            'alert_name': alert_config.name,
            'metric_type': alert_config.metric_type,
            'threshold': alert_config.threshold,
            'current_value': metric_value,
            'message': f"Alert '{alert_config.name}' triggered: {alert_config.metric_type} = {metric_value} (threshold: {alert_config.threshold})"
        }
        
        # Just store the alert data without creating a trace to avoid recursion
        # In a real system, this would send to an external alerting service
        if not hasattr(self, '_triggered_alerts'):
            self._triggered_alerts = []
        self._triggered_alerts.append(alert_data)
    
    def _matches_filters(self, trace_info: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a trace matches the given filters.
        
        Args:
            trace_info: Trace information to check
            filters: Filter criteria
            
        Returns:
            True if trace matches filters
        """
        for key, value in filters.items():
            if key == 'operation_name' and trace_info.get('operation_name') != value:
                return False
            elif key == 'status' and trace_info.get('status') != value:
                return False
            elif key in trace_info.get('metadata', {}) and trace_info['metadata'][key] != value:
                return False
        
        return True