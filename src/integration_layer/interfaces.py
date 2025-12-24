"""Base interfaces for the Integration Layer component."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import asyncio


class HealthStatus(Enum):
    """Health status of external services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class AuthConfig:
    """Authentication configuration for external services."""
    auth_type: str  # 'api_key', 'oauth', 'basic', 'bearer'
    credentials: Dict[str, str]
    refresh_token_url: Optional[str] = None
    token_expiry_buffer: int = 300  # seconds before expiry to refresh


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: float
    burst_size: int
    backoff_factor: float = 2.0
    max_backoff: float = 60.0


@dataclass
class RetryConfig:
    """Retry configuration with exponential backoff."""
    max_attempts: int
    initial_delay: float
    max_delay: float
    backoff_factor: float
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int
    recovery_timeout: float
    success_threshold: int  # for half-open state


@dataclass
class ServiceConfig:
    """Configuration for external service integration."""
    service_name: str
    base_url: str
    auth_config: AuthConfig
    rate_limit_config: RateLimitConfig
    retry_config: RetryConfig
    circuit_breaker_config: CircuitBreakerConfig
    timeout: float = 30.0
    health_check_endpoint: Optional[str] = None
    health_check_interval: float = 60.0


@dataclass
class ServiceHealth:
    """Health status of an external service."""
    service_name: str
    status: HealthStatus
    last_check: datetime
    response_time: Optional[float]
    error_message: Optional[str]
    consecutive_failures: int
    uptime_percentage: float


@dataclass
class IntegrationResult:
    """Result of an integration operation."""
    success: bool
    data: Any
    status_code: Optional[int]
    response_time: float
    attempts: int
    error_message: Optional[str]
    metadata: Dict[str, Any]


class IntegrationLayer(ABC):
    """Abstract base class for integration layer."""
    
    @abstractmethod
    async def register_service(self, config: ServiceConfig) -> bool:
        """Register a new external service."""
        pass
    
    @abstractmethod
    async def unregister_service(self, service_name: str) -> bool:
        """Unregister an external service."""
        pass
    
    @abstractmethod
    async def make_request(self, service_name: str, endpoint: str, method: str = 'GET',
                          data: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None) -> IntegrationResult:
        """Make a request to an external service with retry and rate limiting."""
        pass
    
    @abstractmethod
    async def validate_data_consistency(self, service_name: str, data: Any,
                                      validation_rules: List[Callable]) -> bool:
        """Validate data consistency for inter-service communication."""
        pass
    
    @abstractmethod
    async def update_service_config(self, service_name: str, config: ServiceConfig) -> bool:
        """Update service configuration without restart (hot configuration)."""
        pass
    
    @abstractmethod
    async def get_service_health(self, service_name: str) -> ServiceHealth:
        """Get health status of a specific service."""
        pass
    
    @abstractmethod
    async def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all registered services."""
        pass
    
    @abstractmethod
    async def start_health_monitoring(self) -> None:
        """Start continuous health monitoring for all services."""
        pass
    
    @abstractmethod
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        pass
    
    @abstractmethod
    def get_circuit_breaker_state(self, service_name: str) -> CircuitBreakerState:
        """Get current circuit breaker state for a service."""
        pass
    
    @abstractmethod
    async def reset_circuit_breaker(self, service_name: str) -> bool:
        """Manually reset circuit breaker for a service."""
        pass