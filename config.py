"""
Configuration Management for HackRX 6.0 System
===============================================

This module manages all configuration settings for the intelligent query-retrieval system.
It provides a centralized configuration class with environment variable support and
sensible defaults for development and production environments.

Features:
- Environment variable loading with .env file support
- Type conversion and validation
- Backwards compatibility for legacy imports
- Performance tuning parameters
- Security configuration

Author: HackRX 6.0 Team
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Centralized configuration class for the HackRX application.
    
    This class provides a clean interface to all configuration settings,
    with automatic type conversion and sensible defaults for all environments.
    """
    
    # =============================================================================
    # Environment Configuration
    # =============================================================================
    
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug_mode: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # =============================================================================
    # API Authentication Configuration
    # =============================================================================
    
    # HackRX 6.0 official bearer token (as specified in problem statement)
    api_bearer_token: str = os.getenv(
        "API_BEARER_TOKEN", 
        "e4b975d68599b231b42b0b2face528c5d0df07c55c976fd98c8ab740a50ad638"
    )
    
    # =============================================================================
    # Vector Database Configuration (Pinecone)
    # =============================================================================
    
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "hackrx-intelligent-query-system")
    
    # Vector configuration
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # all-MiniLM-L6-v2
    similarity_metric: str = os.getenv("SIMILARITY_METRIC", "cosine")
    
    # =============================================================================
    # LLM Configuration
    # =============================================================================
    
    # Primary LLM provider: "google" (recommended) or "openai"
    llm_provider: str = os.getenv("LLM_PROVIDER", "google")
    
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # Model selection
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-pro")
    
    # =============================================================================
    # Database Configuration (PostgreSQL)
    # =============================================================================
    
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://hackrx_user:hackrx_password@localhost:5432/hackrx_db"
    )
    
    # Individual database components (for manual construction)
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "hackrx_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "hackrx_password")
    postgres_db: str = os.getenv("POSTGRES_DB", "hackrx_db")
    
    # Connection pool settings
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    
    # =============================================================================
    # Document Processing Configuration - MAXIMUM SPEED OPTIMIZATION
    # =============================================================================
    
    # Text chunking parameters - SPEED OPTIMIZED (fewer, larger chunks)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "4000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "25"))
    
    # Supported document formats
    supported_formats: list = ["pdf", "docx", "eml", "msg", "mbox"]
    
    # Processing limits
    max_document_size_mb: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50"))
    max_pages_per_document: int = int(os.getenv("MAX_PAGES_PER_DOCUMENT", "500"))
    
    # =============================================================================
    # RAG Performance Configuration - SPEED OPTIMIZED
    # =============================================================================
    
    # Retrieval parameters - OPTIMIZED FOR <15s RESPONSE
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "3"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Context management - OPTIMIZED FOR SPEED
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "2500"))
    context_compression_enabled: bool = os.getenv("CONTEXT_COMPRESSION", "true").lower() == "true"
    
    # =============================================================================
    # Token Optimization Configuration
    # =============================================================================
    
    # Token limits and counting
    max_tokens_per_request: int = int(os.getenv("MAX_TOKENS_PER_REQUEST", "2000"))
    enable_token_counting: bool = os.getenv("ENABLE_TOKEN_COUNTING", "true").lower() == "true"
    
    # Caching optimization - IMPORTANT: Caching ONLY helps repeat queries
    # It does NOT improve first-run speed and can add slight overhead
    # Disabled by default for optimal first-response performance
    enable_caching: bool = os.getenv("ENABLE_CACHING", "false").lower() == "true"
    cache_ttl_hours: int = int(os.getenv("CACHE_TTL_HOURS", "1"))
    
    # =============================================================================
    # Performance & Concurrency Configuration - SPEED OPTIMIZED
    # =============================================================================
    
    # Concurrent processing - Reduced for better individual response times
    max_concurrent_questions: int = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "5"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
    
    # Rate limiting - Increased for better throughput
    rate_limit_requests_per_minute: int = int(os.getenv("RATE_LIMIT_RPM", "200"))
    
    # =============================================================================
    # Logging & Monitoring Configuration - MINIMAL FOR SPEED
    # =============================================================================
    
    # Reduced logging for better performance
    log_level: str = os.getenv("LOG_LEVEL", "WARNING")  # Less verbose for speed
    enable_query_logging: bool = os.getenv("ENABLE_QUERY_LOGGING", "false").lower() == "true"  # Disabled for speed
    enable_performance_metrics: bool = os.getenv("ENABLE_PERFORMANCE_METRICS", "true").lower() == "true"
    
    # =============================================================================
    # Utility Methods
    # =============================================================================
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    def get_database_url(self) -> str:
        """
        Get the complete database URL.
        
        Returns:
            str: Complete PostgreSQL connection URL
        """
        if self.database_url and self.database_url != "postgresql://hackrx_user:hackrx_password@localhost:5432/hackrx_db":
            return self.database_url
        
        # Construct from individual components
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    def validate_configuration(self) -> bool:
        """
        Validate critical configuration settings.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If critical settings are missing
        """
        required_settings = []
        
        # Check LLM configuration
        if self.llm_provider == "openai" and not self.openai_api_key:
            required_settings.append("OPENAI_API_KEY")
        elif self.llm_provider == "google" and not self.google_api_key:
            required_settings.append("GOOGLE_API_KEY")
        
        # Check Pinecone configuration (optional but recommended)
        if not self.pinecone_api_key:
            print("Warning: PINECONE_API_KEY not set - vector search will use fallback")
        
        if required_settings:
            raise ValueError(f"Missing required configuration: {', '.join(required_settings)}")
        
        return True


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Create global configuration instance for easy access
config = Config()

# =============================================================================
# Legacy Compatibility Layer
# =============================================================================
# Maintained for backwards compatibility with existing code

API_BEARER_TOKEN = config.api_bearer_token
PINECONE_API_KEY = config.pinecone_api_key
PINECONE_ENVIRONMENT = config.pinecone_environment
PINECONE_INDEX_NAME = config.pinecone_index_name
DATABASE_URL = config.database_url
POSTGRES_USER = config.postgres_user
POSTGRES_PASSWORD = config.postgres_password
POSTGRES_DB = config.postgres_db
POSTGRES_HOST = config.postgres_host
POSTGRES_PORT = config.postgres_port
LLM_PROVIDER = config.llm_provider
OPENAI_API_KEY = config.openai_api_key
GOOGLE_API_KEY = config.google_api_key
CHUNK_SIZE = config.chunk_size
CHUNK_OVERLAP = config.chunk_overlap
TOP_K_RETRIEVAL = config.top_k_retrieval
MAX_CONTEXT_LENGTH = config.max_context_length
MAX_TOKENS_PER_REQUEST = config.max_tokens_per_request
ENABLE_TOKEN_COUNTING = config.enable_token_counting
