from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST, generate_latest
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
import time
from typing import Callable
import logging

logger = logging.getLogger(__name__)

# Create custom registry
REGISTRY = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

ACTIVE_CONNECTIONS = Gauge(
    'http_active_connections',
    'Number of active HTTP connections',
    registry=REGISTRY
)

ML_PREDICTIONS = Counter(
    'ml_predictions_total',
    'Total ML model predictions',
    ['model_name', 'prediction_type'],
    registry=REGISTRY
)

ML_PREDICTION_DURATION = Histogram(
    'ml_prediction_duration_seconds',
    'ML prediction duration in seconds',
    ['model_name'],
    registry=REGISTRY
)

ANALYSIS_ISSUES_FOUND = Counter(
    'analysis_issues_found_total',
    'Total issues found during analysis',
    ['issue_type', 'severity'],
    registry=REGISTRY
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=REGISTRY
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result'],
    registry=REGISTRY
)

def setup_metrics(app: FastAPI):
    """Setup Prometheus metrics for FastAPI app"""
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        """Middleware to collect HTTP metrics"""
        
        # Start timer
        start_time = time.time()
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract metrics labels
            method = request.method
            endpoint = request.url.path
            status_code = response.status_code
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            return response
            
        except Exception as e:
            # Record error
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).inc()
            
            logger.error(f"Request failed: {e}")
            raise
            
        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()
    
    @app.get("/metrics")
    async def get_metrics():
        """Endpoint to expose Prometheus metrics"""
        metrics_data = generate_latest(REGISTRY)
        return PlainTextResponse(
            content=metrics_data.decode('utf-8'),
            media_type=CONTENT_TYPE_LATEST
        )

def record_ml_prediction(model_name: str, prediction_type: str, duration: float):
    """Record ML model prediction metrics"""
    ML_PREDICTIONS.labels(
        model_name=model_name,
        prediction_type=prediction_type
    ).inc()
    
    ML_PREDICTION_DURATION.labels(
        model_name=model_name
    ).observe(duration)

def record_analysis_issue(issue_type: str, severity: str):
    """Record analysis issue metrics"""
    ANALYSIS_ISSUES_FOUND.labels(
        issue_type=issue_type,
        severity=severity
    ).inc()

def record_cache_operation(operation: str, result: str):
    """Record cache operation metrics"""
    CACHE_OPERATIONS.labels(
        operation=operation,
        result=result
    ).inc()

def update_database_connections(count: int):
    """Update active database connections count"""
    DATABASE_CONNECTIONS.set(count)