from pydantic import BaseSettings, Field, validator
from typing import List, Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Basic app settings
    APP_NAME: str = "AI Code Review System"
    DEBUG: bool = Field(default=False, env="DEBUG")
    API_V1_STR: str = "/api/v1"
    
    # Security settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Database settings
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # ML Model settings
    ML_MODEL_PATH: str = Field(default="./models", env="ML_MODEL_PATH")
    ML_BATCH_SIZE: int = Field(default=32, env="ML_BATCH_SIZE")
    ML_MAX_FILE_SIZE: int = Field(default=1024*1024, env="ML_MAX_FILE_SIZE")  # 1MB
    
    # GitHub integration
    GITHUB_WEBHOOK_SECRET: Optional[str] = Field(default=None, env="GITHUB_WEBHOOK_SECRET")
    GITHUB_APP_ID: Optional[str] = Field(default=None, env="GITHUB_APP_ID")
    GITHUB_PRIVATE_KEY: Optional[str] = Field(default=None, env="GITHUB_PRIVATE_KEY")
    
    # Analysis thresholds
    VULNERABILITY_THRESHOLD: float = Field(default=0.7, env="VULNERABILITY_THRESHOLD")
    QUALITY_THRESHOLD: float = Field(default=0.6, env="QUALITY_THRESHOLD")
    COMPLEXITY_THRESHOLD: int = Field(default=10, env="COMPLEXITY_THRESHOLD")
    
    # Celery settings (for background tasks)
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/1", env="CELERY_RESULT_BACKEND")
    
    # Monitoring settings
    PROMETHEUS_METRICS_PORT: int = Field(default=8001, env="PROMETHEUS_METRICS_PORT")
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_allowed_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://')):
            raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()