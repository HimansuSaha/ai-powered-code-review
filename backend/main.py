from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Optional

from app.core.config import get_settings
from app.core.database import init_db
from app.api import auth, analysis, repositories, webhooks, dashboard
from app.core.logger import setup_logging
from app.core.metrics import setup_metrics
from app.ml_models.model_manager import ModelManager

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting AI Code Review System")
    
    # Initialize database
    await init_db()
    
    # Initialize ML models
    model_manager = ModelManager()
    await model_manager.initialize_models()
    app.state.model_manager = model_manager
    
    # Setup metrics
    setup_metrics(app)
    
    logger.info("Application startup complete")
    yield
    
    logger.info("Shutting down AI Code Review System")

# Create FastAPI app
app = FastAPI(
    title="AI Code Review System",
    description="AI-powered code analysis with security vulnerability detection and quality assessment",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Code Analysis"])
app.include_router(repositories.router, prefix="/api/v1/repositories", tags=["Repositories"])
app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["Webhooks"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "AI Code Review System API",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "AI-powered vulnerability detection",
            "Code quality analysis",
            "Real-time GitHub integration",
            "Automated security scanning",
            "Performance metrics"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
    }

@app.get("/api/v1/status")
async def system_status():
    """Detailed system status"""
    try:
        model_manager = app.state.model_manager
        models_status = await model_manager.get_models_status()
        
        return {
            "system": "operational",
            "models": models_status,
            "features": {
                "vulnerability_detection": True,
                "quality_analysis": True,
                "github_integration": True,
                "real_time_analysis": True
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4
    )