"""
Service health monitoring and metrics collection.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import os

logger = logging.getLogger(__name__)


class ServiceMonitor:
    """Monitor service health and performance metrics."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
    
    def record_request(self, response_time: float, error: bool = False):
        """Record a request with its response time."""
        self.request_count += 1
        self.total_response_time += response_time
        if error:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics."""
        uptime = time.time() - self.start_time
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Process specific
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        metrics = {
            "service": self.service_name,
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "requests": {
                "total": self.request_count,
                "errors": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "avg_response_time_ms": (self.total_response_time / max(self.request_count, 1)) * 1000 if self.request_count > 0 else 0
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total_mb": memory.total / (1024 * 1024),
                "memory_available_mb": memory.available / (1024 * 1024),
                "memory_percent": memory.percent,
                "process_memory_mb": process_memory
            }
        }
        
        return metrics
    
    def is_healthy(self) -> bool:
        """Check if service is healthy based on metrics."""
        metrics = self.get_metrics()
        
        # Check error rate
        if metrics["requests"]["error_rate"] > 0.1:  # More than 10% errors
            return False
        
        # Check memory
        if metrics["system"]["memory_percent"] > 90:
            return False
        
        # Check CPU
        if metrics["system"]["cpu_percent"] > 95:
            return False
        
        return True


# Request timing decorator
def monitor_request(monitor: ServiceMonitor):
    """Decorator to monitor request timing and errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = False
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = True
                raise
            finally:
                response_time = time.time() - start_time
                monitor.record_request(response_time, error)
        
        return wrapper
    return decorator


# Global monitors for each service (initialized in main.py)
_service_monitors: Dict[str, ServiceMonitor] = {}


def get_monitor(service_name: str) -> ServiceMonitor:
    """Get or create a monitor for a service."""
    if service_name not in _service_monitors:
        _service_monitors[service_name] = ServiceMonitor(service_name)
    return _service_monitors[service_name]


def get_all_metrics() -> Dict[str, Any]:
    """Get metrics from all monitored services."""
    return {
        name: monitor.get_metrics() 
        for name, monitor in _service_monitors.items()
    }
