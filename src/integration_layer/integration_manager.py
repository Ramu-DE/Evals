"""Implementation of the Integration Layer for managing external API connections."""

import asyncio
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import aiohttp
from dataclasses import replace

from .interfaces import (
    IntegrationLayer,
    ServiceConfig,
    ServiceHealth,
    IntegrationResult,
    HealthStatus,
    CircuitBreakerState,
    AuthConfig,
    RateLimitConfig,
    RetryConfig,
    CircuitBreakerConfig
)


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a token for rate limiting."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.config.burst_size,
                self.tokens + elapsed * self.config.requests_per_second
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    async def wait_for_token(self) -> None:
        """Wait until a token is available."""
        while not await self.acquire():
            await asyncio.sleep(1.0 / self.config.requests_per_second)


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if (time.time() - self.last_failure_time) > self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception(f"Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state
    
    async def reset(self):
        """Reset circuit breaker to closed state."""
        async with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None


class IntegrationManager(IntegrationLayer):
    """Implementation of the Integration Layer."""
    
    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.auth_tokens: Dict[str, Dict[str, Any]] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        if self.monitoring_task:
            self.monitoring_task.cancel()
    
    async def register_service(self, config: ServiceConfig) -> bool:
        """Register a new external service."""
        try:
            self.services[config.service_name] = config
            self.rate_limiters[config.service_name] = RateLimiter(config.rate_limit_config)
            self.circuit_breakers[config.service_name] = CircuitBreaker(config.circuit_breaker_config)
            
            # Initialize service health
            self.service_health[config.service_name] = ServiceHealth(
                service_name=config.service_name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                response_time=None,
                error_message=None,
                consecutive_failures=0,
                uptime_percentage=0.0
            )
            
            # Initialize authentication if needed
            await self._initialize_auth(config)
            
            self.logger.info(f"Registered service: {config.service_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register service {config.service_name}: {e}")
            return False
    
    async def unregister_service(self, service_name: str) -> bool:
        """Unregister an external service."""
        try:
            if service_name in self.services:
                del self.services[service_name]
                del self.rate_limiters[service_name]
                del self.circuit_breakers[service_name]
                del self.service_health[service_name]
                if service_name in self.auth_tokens:
                    del self.auth_tokens[service_name]
                
                self.logger.info(f"Unregistered service: {service_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to unregister service {service_name}: {e}")
            return False
    
    async def make_request(self, service_name: str, endpoint: str, method: str = 'GET',
                          data: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None) -> IntegrationResult:
        """Make a request to an external service with retry and rate limiting."""
        if service_name not in self.services:
            return IntegrationResult(
                success=False,
                data=None,
                status_code=None,
                response_time=0.0,
                attempts=0,
                error_message=f"Service {service_name} not registered",
                metadata={}
            )
        
        config = self.services[service_name]
        rate_limiter = self.rate_limiters[service_name]
        circuit_breaker = self.circuit_breakers[service_name]
        
        start_time = time.time()
        attempts = 0
        last_error = None
        
        for attempt in range(config.retry_config.max_attempts):
            attempts += 1
            
            try:
                # Rate limiting
                await rate_limiter.wait_for_token()
                
                # Circuit breaker protection
                result = await circuit_breaker.call(
                    self._execute_request,
                    config, endpoint, method, data, headers
                )
                
                response_time = time.time() - start_time
                return IntegrationResult(
                    success=True,
                    data=result['data'],
                    status_code=result['status_code'],
                    response_time=response_time,
                    attempts=attempts,
                    error_message=None,
                    metadata=result.get('metadata', {})
                )
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Request attempt {attempt + 1} failed for {service_name}: {e}")
                
                if attempt < config.retry_config.max_attempts - 1:
                    delay = min(
                        config.retry_config.initial_delay * (config.retry_config.backoff_factor ** attempt),
                        config.retry_config.max_delay
                    )
                    
                    if config.retry_config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    await asyncio.sleep(delay)
        
        response_time = time.time() - start_time
        return IntegrationResult(
            success=False,
            data=None,
            status_code=None,
            response_time=response_time,
            attempts=attempts,
            error_message=last_error,
            metadata={}
        )
    
    async def _execute_request(self, config: ServiceConfig, endpoint: str, method: str,
                              data: Optional[Dict[str, Any]], headers: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Execute the actual HTTP request."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        request_headers = headers or {}
        
        # Add authentication headers
        auth_headers = await self._get_auth_headers(config)
        request_headers.update(auth_headers)
        
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        
        async with self.session.request(
            method=method,
            url=url,
            json=data if method.upper() in ['POST', 'PUT', 'PATCH'] else None,
            params=data if method.upper() == 'GET' else None,
            headers=request_headers,
            timeout=timeout
        ) as response:
            response_data = await response.text()
            
            if response.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=response_data
                )
            
            try:
                json_data = await response.json()
            except:
                json_data = response_data
            
            return {
                'data': json_data,
                'status_code': response.status,
                'metadata': {
                    'headers': dict(response.headers),
                    'url': str(response.url)
                }
            }
    
    async def validate_data_consistency(self, service_name: str, data: Any,
                                      validation_rules: List[Callable]) -> bool:
        """Validate data consistency for inter-service communication."""
        try:
            for rule in validation_rules:
                try:
                    if asyncio.iscoroutinefunction(rule):
                        result = await rule(data)
                    else:
                        result = rule(data)
                    
                    if not result:
                        self.logger.warning(f"Data validation failed for {service_name}")
                        return False
                except Exception as rule_error:
                    self.logger.warning(f"Data validation rule failed for {service_name}: {rule_error}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Data validation error for {service_name}: {e}")
            return False
    
    async def update_service_config(self, service_name: str, config: ServiceConfig) -> bool:
        """Update service configuration without restart (hot configuration)."""
        try:
            if service_name not in self.services:
                return False
            
            old_config = self.services[service_name]
            self.services[service_name] = config
            
            # Update rate limiter if config changed
            if old_config.rate_limit_config != config.rate_limit_config:
                self.rate_limiters[service_name] = RateLimiter(config.rate_limit_config)
            
            # Update circuit breaker if config changed
            if old_config.circuit_breaker_config != config.circuit_breaker_config:
                self.circuit_breakers[service_name] = CircuitBreaker(config.circuit_breaker_config)
            
            # Update authentication if changed
            if old_config.auth_config != config.auth_config:
                await self._initialize_auth(config)
            
            self.logger.info(f"Updated configuration for service: {service_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update config for {service_name}: {e}")
            return False
    
    async def get_service_health(self, service_name: str) -> ServiceHealth:
        """Get health status of a specific service."""
        if service_name not in self.service_health:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                response_time=None,
                error_message="Service not registered",
                consecutive_failures=0,
                uptime_percentage=0.0
            )
        
        return self.service_health[service_name]
    
    async def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all registered services."""
        return self.service_health.copy()
    
    async def start_health_monitoring(self) -> None:
        """Start continuous health monitoring for all services."""
        if self.monitoring_task and not self.monitoring_task.done():
            return
        
        self.monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        self.logger.info("Started health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        self.logger.info("Stopped health monitoring")
    
    def get_circuit_breaker_state(self, service_name: str) -> CircuitBreakerState:
        """Get current circuit breaker state for a service."""
        if service_name not in self.circuit_breakers:
            return CircuitBreakerState.CLOSED
        return self.circuit_breakers[service_name].get_state()
    
    async def reset_circuit_breaker(self, service_name: str) -> bool:
        """Manually reset circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            return False
        
        await self.circuit_breakers[service_name].reset()
        self.logger.info(f"Reset circuit breaker for service: {service_name}")
        return True
    
    async def _initialize_auth(self, config: ServiceConfig) -> None:
        """Initialize authentication for a service."""
        if config.auth_config.auth_type == 'oauth':
            await self._refresh_oauth_token(config)
    
    async def _get_auth_headers(self, config: ServiceConfig) -> Dict[str, str]:
        """Get authentication headers for a service."""
        headers = {}
        auth_config = config.auth_config
        
        if auth_config.auth_type == 'api_key':
            api_key = auth_config.credentials.get('api_key', '')
            if api_key:
                headers['Authorization'] = f"Bearer {api_key}"
        elif auth_config.auth_type == 'bearer':
            token = auth_config.credentials.get('token', '')
            if token:
                headers['Authorization'] = f"Bearer {token}"
        elif auth_config.auth_type == 'basic':
            import base64
            username = auth_config.credentials.get('username', '')
            password = auth_config.credentials.get('password', '')
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers['Authorization'] = f"Basic {credentials}"
        elif auth_config.auth_type == 'oauth':
            token_info = self.auth_tokens.get(config.service_name, {})
            access_token = token_info.get('access_token')
            if access_token:
                headers['Authorization'] = f"Bearer {access_token}"
        
        return headers
    
    async def _refresh_oauth_token(self, config: ServiceConfig) -> None:
        """Refresh OAuth token if needed."""
        # Implementation would depend on specific OAuth flow
        # This is a placeholder for OAuth token refresh logic
        pass
    
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        while True:
            try:
                for service_name, config in self.services.items():
                    await self._check_service_health(service_name, config)
                
                # Wait for the shortest health check interval
                min_interval = min(
                    (config.health_check_interval for config in self.services.values()),
                    default=60.0
                )
                await asyncio.sleep(min_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10.0)  # Wait before retrying
    
    async def _check_service_health(self, service_name: str, config: ServiceConfig) -> None:
        """Check health of a specific service."""
        current_health = self.service_health[service_name]
        
        # Skip if recently checked
        if (datetime.now() - current_health.last_check).total_seconds() < config.health_check_interval:
            return
        
        start_time = time.time()
        
        try:
            if config.health_check_endpoint:
                result = await self.make_request(service_name, config.health_check_endpoint, 'GET')
                
                if result.success:
                    status = HealthStatus.HEALTHY
                    error_message = None
                    consecutive_failures = 0
                else:
                    status = HealthStatus.UNHEALTHY
                    error_message = result.error_message
                    consecutive_failures = current_health.consecutive_failures + 1
            else:
                # Basic connectivity check
                status = HealthStatus.HEALTHY
                error_message = None
                consecutive_failures = 0
            
            response_time = time.time() - start_time
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            error_message = str(e)
            consecutive_failures = current_health.consecutive_failures + 1
            response_time = time.time() - start_time
        
        # Calculate uptime percentage (simplified)
        uptime_percentage = max(0.0, 100.0 - (consecutive_failures * 10.0))
        
        self.service_health[service_name] = ServiceHealth(
            service_name=service_name,
            status=status,
            last_check=datetime.now(),
            response_time=response_time,
            error_message=error_message,
            consecutive_failures=consecutive_failures,
            uptime_percentage=uptime_percentage
        )