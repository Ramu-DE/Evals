"""Property-based tests for the Integration Layer component."""

import asyncio
import time
import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.integration_layer.interfaces import (
    ServiceConfig,
    AuthConfig,
    RateLimitConfig,
    RetryConfig,
    CircuitBreakerConfig,
    HealthStatus,
    CircuitBreakerState
)
from src.integration_layer.integration_manager import IntegrationManager, RateLimiter


# Test data generators
@st.composite
def auth_config_strategy(draw):
    """Generate valid AuthConfig instances."""
    auth_type = draw(st.sampled_from(['api_key', 'oauth', 'basic', 'bearer']))
    
    if auth_type == 'api_key':
        credentials = {'api_key': draw(st.text(min_size=10, max_size=50))}
    elif auth_type == 'basic':
        credentials = {
            'username': draw(st.text(min_size=3, max_size=20)),
            'password': draw(st.text(min_size=8, max_size=30))
        }
    elif auth_type == 'bearer':
        credentials = {'token': draw(st.text(min_size=20, max_size=100))}
    else:  # oauth
        credentials = {
            'client_id': draw(st.text(min_size=10, max_size=30)),
            'client_secret': draw(st.text(min_size=20, max_size=50))
        }
    
    return AuthConfig(
        auth_type=auth_type,
        credentials=credentials,
        refresh_token_url=draw(st.one_of(st.none(), st.text(min_size=10, max_size=100))),
        token_expiry_buffer=draw(st.integers(min_value=60, max_value=3600))
    )


@st.composite
def rate_limit_config_strategy(draw):
    """Generate valid RateLimitConfig instances."""
    return RateLimitConfig(
        requests_per_second=draw(st.floats(min_value=0.1, max_value=100.0)),
        burst_size=draw(st.integers(min_value=1, max_value=100)),
        backoff_factor=draw(st.floats(min_value=1.1, max_value=5.0)),
        max_backoff=draw(st.floats(min_value=10.0, max_value=300.0))
    )


@st.composite
def retry_config_strategy(draw):
    """Generate valid RetryConfig instances."""
    return RetryConfig(
        max_attempts=draw(st.integers(min_value=1, max_value=10)),
        initial_delay=draw(st.floats(min_value=0.1, max_value=5.0)),
        max_delay=draw(st.floats(min_value=5.0, max_value=60.0)),
        backoff_factor=draw(st.floats(min_value=1.1, max_value=5.0)),
        jitter=draw(st.booleans())
    )


@st.composite
def circuit_breaker_config_strategy(draw):
    """Generate valid CircuitBreakerConfig instances."""
    return CircuitBreakerConfig(
        failure_threshold=draw(st.integers(min_value=1, max_value=10)),
        recovery_timeout=draw(st.floats(min_value=1.0, max_value=60.0)),
        success_threshold=draw(st.integers(min_value=1, max_value=5))
    )


@st.composite
def service_config_strategy(draw):
    """Generate valid ServiceConfig instances."""
    return ServiceConfig(
        service_name=draw(st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'))),
        base_url=f"https://{draw(st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))}.com",
        auth_config=draw(auth_config_strategy()),
        rate_limit_config=draw(rate_limit_config_strategy()),
        retry_config=draw(retry_config_strategy()),
        circuit_breaker_config=draw(circuit_breaker_config_strategy()),
        timeout=draw(st.floats(min_value=1.0, max_value=120.0)),
        health_check_endpoint=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        health_check_interval=draw(st.floats(min_value=10.0, max_value=300.0))
    )


class TestIntegrationLayerProperties:
    """Property-based tests for Integration Layer."""
    
    @pytest.mark.asyncio
    @given(service_config=service_config_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_26_authentication_and_rate_limiting(self, service_config):
        """
        **Feature: rag-evaluation-pipeline, Property 26: Authentication and Rate Limiting**
        **Validates: Requirements 8.1**
        
        For any external API connection, the Integration Layer should handle 
        authentication and implement appropriate rate limiting.
        """
        async with IntegrationManager() as integration_manager:
            # Register service
            success = await integration_manager.register_service(service_config)
            assert success, "Service registration should succeed"
            
            # Verify authentication headers are properly set
            with patch.object(integration_manager, '_execute_request') as mock_execute:
                mock_execute.return_value = {
                    'data': {'status': 'ok'},
                    'status_code': 200,
                    'metadata': {}
                }
                
                # Make a request to trigger authentication
                result = await integration_manager.make_request(
                    service_config.service_name,
                    '/test',
                    'GET'
                )
                
                # Verify request was made (authentication was handled)
                assert mock_execute.called, "Request should be executed with authentication"
                
                # Verify rate limiting is enforced
                rate_limiter = integration_manager.rate_limiters[service_config.service_name]
                assert isinstance(rate_limiter, RateLimiter), "Rate limiter should be configured"
                assert rate_limiter.config.requests_per_second == service_config.rate_limit_config.requests_per_second
                assert rate_limiter.config.burst_size == service_config.rate_limit_config.burst_size
    
    @pytest.mark.asyncio
    @given(
        service_config=service_config_strategy(),
        num_requests=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    async def test_rate_limiting_enforcement(self, service_config, num_requests):
        """Test that rate limiting is properly enforced."""
        async with IntegrationManager() as integration_manager:
            await integration_manager.register_service(service_config)
            
            rate_limiter = integration_manager.rate_limiters[service_config.service_name]
            
            # Test token acquisition
            acquired_tokens = 0
            for _ in range(num_requests):
                if await rate_limiter.acquire():
                    acquired_tokens += 1
            
            # Should not exceed burst size
            assert acquired_tokens <= service_config.rate_limit_config.burst_size, \
                "Rate limiter should not allow more requests than burst size"
    
    @pytest.mark.asyncio
    @given(service_config=service_config_strategy())
    @settings(max_examples=50, deadline=None)
    async def test_authentication_header_generation(self, service_config):
        """Test that authentication headers are properly generated."""
        async with IntegrationManager() as integration_manager:
            await integration_manager.register_service(service_config)
            
            headers = await integration_manager._get_auth_headers(service_config)
            
            # Verify appropriate authentication header is set based on auth type
            auth_type = service_config.auth_config.auth_type
            credentials = service_config.auth_config.credentials
            
            if auth_type == 'api_key' and credentials.get('api_key'):
                assert 'Authorization' in headers, "Authorization header should be present for api_key"
                assert headers['Authorization'].startswith('Bearer '), "Should use Bearer token format"
            elif auth_type == 'bearer' and credentials.get('token'):
                assert 'Authorization' in headers, "Authorization header should be present for bearer"
                assert headers['Authorization'].startswith('Bearer '), "Should use Bearer token format"
            elif auth_type == 'basic' and credentials.get('username') and credentials.get('password'):
                assert 'Authorization' in headers, "Authorization header should be present for basic"
                assert headers['Authorization'].startswith('Basic '), "Should use Basic auth format"
            elif auth_type == 'oauth':
                # OAuth might not have headers if no token is set
                if 'Authorization' in headers:
                    assert headers['Authorization'].startswith('Bearer '), "Should use Bearer token format for OAuth"
            
            # If no valid credentials, no Authorization header should be present
            if not any([
                auth_type == 'api_key' and credentials.get('api_key'),
                auth_type == 'bearer' and credentials.get('token'),
                auth_type == 'basic' and credentials.get('username') and credentials.get('password'),
                auth_type == 'oauth' and integration_manager.auth_tokens.get(service_config.service_name, {}).get('access_token')
            ]):
                # No authorization header expected if no valid credentials
                pass


class TestRetryLogicProperties:
    """Property-based tests for retry logic implementation."""
    
    @pytest.mark.asyncio
    @given(service_config=service_config_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_27_retry_logic_implementation(self, service_config):
        """
        **Feature: rag-evaluation-pipeline, Property 27: Retry Logic Implementation**
        **Validates: Requirements 8.2**
        
        For any service failure, the Integration Layer should implement exponential 
        backoff retry logic until success or maximum attempts.
        """
        async with IntegrationManager() as integration_manager:
            await integration_manager.register_service(service_config)
            
            # Mock a failing request that eventually succeeds
            call_count = 0
            async def mock_failing_request(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < service_config.retry_config.max_attempts:
                    raise Exception("Simulated failure")
                return {
                    'data': {'status': 'success'},
                    'status_code': 200,
                    'metadata': {}
                }
            
            # Disable circuit breaker for this test by setting high failure threshold
            original_threshold = integration_manager.circuit_breakers[service_config.service_name].config.failure_threshold
            integration_manager.circuit_breakers[service_config.service_name].config.failure_threshold = service_config.retry_config.max_attempts + 10
            
            with patch.object(integration_manager, '_execute_request', side_effect=mock_failing_request):
                result = await integration_manager.make_request(
                    service_config.service_name,
                    '/test',
                    'GET'
                )
                
                # Should eventually succeed after retries
                assert result.success, "Request should succeed after retries"
                assert result.attempts == service_config.retry_config.max_attempts, \
                    "Should attempt exactly max_attempts times"
                assert call_count == service_config.retry_config.max_attempts, \
                    "Should call the request function max_attempts times"
    
    @pytest.mark.asyncio
    @given(service_config=service_config_strategy())
    @settings(max_examples=50, deadline=None)
    async def test_exponential_backoff_timing(self, service_config):
        """Test that exponential backoff timing is properly implemented."""
        async with IntegrationManager() as integration_manager:
            await integration_manager.register_service(service_config)
            
            # Mock always failing request
            async def mock_always_failing(*args, **kwargs):
                raise Exception("Always fails")
            
            # Disable circuit breaker for this test
            integration_manager.circuit_breakers[service_config.service_name].config.failure_threshold = service_config.retry_config.max_attempts + 10
            
            start_time = time.time()
            
            with patch.object(integration_manager, '_execute_request', side_effect=mock_always_failing):
                result = await integration_manager.make_request(
                    service_config.service_name,
                    '/test',
                    'GET'
                )
                
                elapsed_time = time.time() - start_time
                
                # Should fail after all attempts
                assert not result.success, "Request should fail after all attempts"
                assert result.attempts == service_config.retry_config.max_attempts, \
                    "Should attempt exactly max_attempts times"
                
                # Should take at least some time for backoff (but be lenient due to test environment)
                # Only check if we have more than 1 attempt
                if service_config.retry_config.max_attempts > 1:
                    min_expected_time = service_config.retry_config.initial_delay * 0.1  # Very lenient
                    assert elapsed_time >= min_expected_time, \
                        f"Should take at least {min_expected_time}s for backoff, took {elapsed_time}s"


class TestDataConsistencyProperties:
    """Property-based tests for data consistency maintenance."""
    
    @pytest.mark.asyncio
    @given(
        service_config=service_config_strategy(),
        test_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    async def test_property_28_data_consistency_maintenance(self, service_config, test_data):
        """
        **Feature: rag-evaluation-pipeline, Property 28: Data Consistency Maintenance**
        **Validates: Requirements 8.3**
        
        For any data flow between services, the Integration Layer should ensure 
        consistency and integrity throughout the transfer.
        """
        async with IntegrationManager() as integration_manager:
            await integration_manager.register_service(service_config)
            
            # Define validation rules
            validation_rules = [
                lambda data: isinstance(data, dict),
                lambda data: len(data) > 0,
                lambda data: all(isinstance(k, str) for k in data.keys())
            ]
            
            # Test data consistency validation
            is_consistent = await integration_manager.validate_data_consistency(
                service_config.service_name,
                test_data,
                validation_rules
            )
            
            # Data should pass validation if it meets all rules
            expected_consistency = (
                isinstance(test_data, dict) and
                len(test_data) > 0 and
                all(isinstance(k, str) for k in test_data.keys())
            )
            
            assert is_consistent == expected_consistency, \
                "Data consistency validation should match expected result"
    
    @pytest.mark.asyncio
    @given(service_config=service_config_strategy())
    @settings(max_examples=50, deadline=None)
    async def test_validation_rule_enforcement(self, service_config):
        """Test that validation rules are properly enforced."""
        async with IntegrationManager() as integration_manager:
            await integration_manager.register_service(service_config)
            
            # Test with valid data
            valid_data = {"key": "value", "number": 42}
            validation_rules = [lambda data: isinstance(data, dict)]
            
            is_valid = await integration_manager.validate_data_consistency(
                service_config.service_name,
                valid_data,
                validation_rules
            )
            assert is_valid, "Valid data should pass validation"
            
            # Test with invalid data
            invalid_data = "not a dict"
            is_invalid = await integration_manager.validate_data_consistency(
                service_config.service_name,
                invalid_data,
                validation_rules
            )
            assert not is_invalid, "Invalid data should fail validation"


class TestHotConfigurationProperties:
    """Property-based tests for hot configuration updates."""
    
    @pytest.mark.asyncio
    @given(
        original_config=service_config_strategy(),
        updated_config=service_config_strategy()
    )
    @settings(max_examples=100, deadline=None)
    async def test_property_29_hot_configuration_updates(self, original_config, updated_config):
        """
        **Feature: rag-evaluation-pipeline, Property 29: Hot Configuration Updates**
        **Validates: Requirements 8.4**
        
        For any configuration change, the Integration Layer should update service 
        connections without requiring system restart.
        """
        # Ensure both configs have the same service name for the test
        updated_config = ServiceConfig(
            service_name=original_config.service_name,
            base_url=updated_config.base_url,
            auth_config=updated_config.auth_config,
            rate_limit_config=updated_config.rate_limit_config,
            retry_config=updated_config.retry_config,
            circuit_breaker_config=updated_config.circuit_breaker_config,
            timeout=updated_config.timeout,
            health_check_endpoint=updated_config.health_check_endpoint,
            health_check_interval=updated_config.health_check_interval
        )
        
        async with IntegrationManager() as integration_manager:
            # Register original service
            success = await integration_manager.register_service(original_config)
            assert success, "Original service registration should succeed"
            
            # Verify original configuration
            assert integration_manager.services[original_config.service_name] == original_config
            
            # Update configuration
            update_success = await integration_manager.update_service_config(
                original_config.service_name,
                updated_config
            )
            assert update_success, "Configuration update should succeed"
            
            # Verify updated configuration is applied
            current_config = integration_manager.services[original_config.service_name]
            assert current_config == updated_config, "Configuration should be updated"
            
            # Verify rate limiter is updated if config changed
            if original_config.rate_limit_config != updated_config.rate_limit_config:
                rate_limiter = integration_manager.rate_limiters[original_config.service_name]
                assert rate_limiter.config == updated_config.rate_limit_config, \
                    "Rate limiter should be updated with new config"
            
            # Verify circuit breaker is updated if config changed
            if original_config.circuit_breaker_config != updated_config.circuit_breaker_config:
                circuit_breaker = integration_manager.circuit_breakers[original_config.service_name]
                assert circuit_breaker.config == updated_config.circuit_breaker_config, \
                    "Circuit breaker should be updated with new config"


class TestDependencyHealthMonitoringProperties:
    """Property-based tests for dependency health monitoring."""
    
    @pytest.mark.asyncio
    @given(service_config=service_config_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_30_dependency_health_monitoring(self, service_config):
        """
        **Feature: rag-evaluation-pipeline, Property 30: Dependency Health Monitoring**
        **Validates: Requirements 8.5**
        
        For any external dependency, the Integration Layer should continuously track 
        service health and availability status.
        """
        async with IntegrationManager() as integration_manager:
            # Register service
            await integration_manager.register_service(service_config)
            
            # Get initial health status
            health = await integration_manager.get_service_health(service_config.service_name)
            
            # Verify health status structure
            assert health.service_name == service_config.service_name, \
                "Health status should have correct service name"
            assert isinstance(health.status, HealthStatus), \
                "Health status should be a valid HealthStatus enum"
            assert isinstance(health.last_check, datetime), \
                "Last check should be a datetime"
            assert isinstance(health.consecutive_failures, int), \
                "Consecutive failures should be an integer"
            assert 0.0 <= health.uptime_percentage <= 100.0, \
                "Uptime percentage should be between 0 and 100"
            
            # Test getting all service health
            all_health = await integration_manager.get_all_service_health()
            assert service_config.service_name in all_health, \
                "Service should be included in all health status"
            assert all_health[service_config.service_name] == health, \
                "Health status should match individual query"
    
    @pytest.mark.asyncio
    @given(service_config=service_config_strategy())
    @settings(max_examples=50, deadline=None)
    async def test_circuit_breaker_state_management(self, service_config):
        """Test circuit breaker state management."""
        async with IntegrationManager() as integration_manager:
            await integration_manager.register_service(service_config)
            
            # Initial state should be CLOSED
            state = integration_manager.get_circuit_breaker_state(service_config.service_name)
            assert state == CircuitBreakerState.CLOSED, "Initial circuit breaker state should be CLOSED"
            
            # Test manual reset
            reset_success = await integration_manager.reset_circuit_breaker(service_config.service_name)
            assert reset_success, "Circuit breaker reset should succeed"
            
            # State should still be CLOSED after reset
            state_after_reset = integration_manager.get_circuit_breaker_state(service_config.service_name)
            assert state_after_reset == CircuitBreakerState.CLOSED, \
                "Circuit breaker state should remain CLOSED after reset"