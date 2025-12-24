"""Configuration settings for RAG Evaluation Pipeline."""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Qdrant Configuration
    qdrant_url: str = "https://55e5707b-2a70-4486-a3cc-25a0b03ade8c.us-east4-0.gcp.cloud.qdrant.io"
    qdrant_api_key: Optional[str] = None
    
    # OPIK Configuration
    opik_api_key: Optional[str] = None
    opik_workspace: Optional[str] = None
    
    # Application Configuration
    app_name: str = "RAG Evaluation Pipeline"
    debug: bool = False
    log_level: str = "INFO"
    
    # Testing Configuration
    hypothesis_max_examples: int = 100
    hypothesis_deadline: int = 5000  # milliseconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()