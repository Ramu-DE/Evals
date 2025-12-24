#!/usr/bin/env python3
"""
Demo script for the Integration Layer component.

This script demonstrates how to use the Integration Layer to manage external API connections
with authentication, rate limiting, retry logic, and health monitoring.
"""

import asyncio
import logging
from src.integration_layer import (
    IntegrationManager,
    ServiceConfig,
    AuthConfig,
    RateLimitConfig,
    RetryConfig,
    CircuitBreakerConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_integration_layer():
    """Demonstrate Integration Layer functionality."""
    
    # Create service configurations
    qdrant_config = ServiceConfig(
        service_name="qdrant",
        base_url="https://55e5707b-2a70-4486-a3cc-25a0b03ade8c.us-east4-0.gcp.cloud.qdrant.io",
        auth_config=AuthConfig(
            auth_type="api_key",
            credentials={"api_key": "your-qdrant-api-key-here"}
        ),
        rate_limit_config=RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20
        ),
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30.0,
            success_threshold=2
        ),
        timeout=30.0,
        health_check_endpoint="/collections",
        health_check_interval=60.0
    )
    
    opik_config = ServiceConfig(
        service_name="opik",
        base_url="https://www.comet.com/opik/api",
        auth_config=AuthConfig(
            auth_type="bearer",
            credentials={"token": "your-opik-token-here"}
        ),
        rate_limit_config=RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10
        ),
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=5.0,
            backoff_factor=2.0
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            success_threshold=1
        ),
        timeout=15.0,
        health_check_endpoint="/health",
        health_check_interval=120.0
    )
    
    async with IntegrationManager() as integration_manager:
        logger.info("Starting Integration Layer Demo")
        
        # Register services
        logger.info("Registering services...")
        await integration_manager.register_service(qdrant_config)
        await integration_manager.register_service(opik_config)
        
        # Start health monitoring
        logger.info("Starting health monitoring...")
        await integration_manager.start_health_monitoring()
        
        # Get service health status
        logger.info("Checking service health...")
        qdrant_health = await integration_manager.get_service_health("qdrant")
        opik_health = await integration_manager.get_service_health("opik")
        
        logger.info(f"Qdrant Health: {qdrant_health.status.value}")
        logger.info(f"OPIK Health: {opik_health.status.value}")
        
        # Get all service health
        all_health = await integration_manager.get_all_service_health()
        logger.info(f"Total registered services: {len(all_health)}")
        
        # Demonstrate data consistency validation
        logger.info("Testing data consistency validation...")
        test_data = {"query": "test", "limit": 10}
        validation_rules = [
            lambda data: isinstance(data, dict),
            lambda data: "query" in data,
            lambda data: isinstance(data.get("limit"), int)
        ]
        
        is_consistent = await integration_manager.validate_data_consistency(
            "qdrant", test_data, validation_rules
        )
        logger.info(f"Data consistency check: {'PASSED' if is_consistent else 'FAILED'}")
        
        # Demonstrate hot configuration update
        logger.info("Testing hot configuration update...")
        updated_config = ServiceConfig(
            service_name="qdrant",
            base_url=qdrant_config.base_url,
            auth_config=qdrant_config.auth_config,
            rate_limit_config=RateLimitConfig(
                requests_per_second=15.0,  # Increased rate limit
                burst_size=30
            ),
            retry_config=qdrant_config.retry_config,
            circuit_breaker_config=qdrant_config.circuit_breaker_config,
            timeout=qdrant_config.timeout,
            health_check_endpoint=qdrant_config.health_check_endpoint,
            health_check_interval=qdrant_config.health_check_interval
        )
        
        update_success = await integration_manager.update_service_config("qdrant", updated_config)
        logger.info(f"Configuration update: {'SUCCESS' if update_success else 'FAILED'}")
        
        # Check circuit breaker states
        logger.info("Checking circuit breaker states...")
        qdrant_cb_state = integration_manager.get_circuit_breaker_state("qdrant")
        opik_cb_state = integration_manager.get_circuit_breaker_state("opik")
        
        logger.info(f"Qdrant Circuit Breaker: {qdrant_cb_state.value}")
        logger.info(f"OPIK Circuit Breaker: {opik_cb_state.value}")
        
        # Demonstrate making requests (would fail without real API keys)
        logger.info("Making test requests...")
        try:
            # This would normally make a real request to Qdrant
            result = await integration_manager.make_request(
                "qdrant",
                "/collections",
                "GET"
            )
            logger.info(f"Qdrant request result: {result.success}")
        except Exception as e:
            logger.info(f"Qdrant request failed (expected without real API key): {e}")
        
        # Stop health monitoring
        logger.info("Stopping health monitoring...")
        await integration_manager.stop_health_monitoring()
        
        logger.info("Integration Layer Demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_integration_layer())