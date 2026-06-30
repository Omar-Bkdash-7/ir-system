"""
Health check utilities for all services.

This module provides a standardized health check endpoint that can be
included in all FastAPI services.
"""

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
import os

router = APIRouter(tags=["health"])

# Service startup time
SERVICE_START_TIME = time.time()


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    service: str
    uptime_seconds: float
    timestamp: float
    version: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(response: Response) -> HealthResponse:
    """
    Health check endpoint for container orchestration.
    
    Returns:
        HealthResponse: Service health information
        
    This endpoint is used by:
    - Docker HEALTHCHECK
    - Kubernetes liveness/readiness probes
    - Load balancers
    - Monitoring systems
    """
    try:
        service_name = os.getenv("SERVICE_NAME", "unknown-service")
        version = os.getenv("SERVICE_VERSION", "1.0.0")
        
        uptime = time.time() - SERVICE_START_TIME
        
        return HealthResponse(
            status="healthy",
            service=service_name,
            uptime_seconds=round(uptime, 2),
            timestamp=time.time(),
            version=version,
            details={
                "environment": os.getenv("ENVIRONMENT", "production"),
                "python_version": os.sys.version.split()[0],
            }
        )
    except Exception as e:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="unhealthy",
            service=os.getenv("SERVICE_NAME", "unknown-service"),
            uptime_seconds=0,
            timestamp=time.time(),
            details={"error": str(e)}
        )


@router.get("/readiness", status_code=status.HTTP_200_OK)
async def readiness_check(response: Response) -> Dict[str, str]:
    """
    Readiness check - indicates if service is ready to handle requests.
    
    Override this in your service to add specific readiness checks
    (e.g., database connection, external API availability).
    """
    # Default: service is ready if it responds
    return {"status": "ready", "service": os.getenv("SERVICE_NAME", "unknown")}


@router.get("/liveness", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check - indicates if service is alive.
    
    This is simpler than health check and just confirms the service is running.
    Used by Kubernetes to restart unhealthy containers.
    """
    return {"status": "alive"}
