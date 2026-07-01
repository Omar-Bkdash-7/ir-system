"""
Redis caching utilities for IR system.
Provides simple caching decorators and functions.
"""

import json
import hashlib
import logging
from typing import Optional, Any, Callable
from functools import wraps
import os

logger = logging.getLogger(__name__)

# Redis client (initialized lazily)
_redis_client = None


def get_redis_client():
    """Get or create Redis client."""
    global _redis_client
    
    if _redis_client is None:
        try:
            import redis
            redis_url = os.getenv("REDIS_URL")
            
            if redis_url:
                # Use REDIS_URL if provided (includes password)
                _redis_client = redis.from_url(redis_url, decode_responses=True)
            else:
                # Fall back to individual components
                redis_host = os.getenv("REDIS_HOST", "redis")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                redis_db = int(os.getenv("REDIS_DB", "0"))
                redis_password = os.getenv("REDIS_PASSWORD")
                
                _redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password if redis_password else None,
                    decode_responses=True
                )
            
            # Test connection
            _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Caching disabled.")
            _redis_client = False  # Mark as unavailable
    
    return _redis_client if _redis_client is not False else None


def is_cache_available() -> bool:
    """Check if Redis cache is available."""
    client = get_redis_client()
    return client is not None


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a consistent cache key from arguments.
    
    Args:
        prefix: Key prefix (e.g., "query", "embedding")
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Cache key string
    """
    # Serialize arguments
    data = {
        "args": args,
        "kwargs": kwargs
    }
    serialized = json.dumps(data, sort_keys=True)
    
    # Hash for consistent key
    hash_value = hashlib.md5(serialized.encode()).hexdigest()
    
    return f"{prefix}:{hash_value}"


def cache_get(key: str) -> Optional[Any]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None
    """
    client = get_redis_client()
    if not client:
        return None
    
    try:
        value = client.get(key)
        if value:
            return json.loads(value)
    except Exception as e:
        logger.error(f"Cache get error: {e}")
    
    return None


def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """
    Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache (must be JSON serializable)
        ttl: Time to live in seconds (None = use default)
    
    Returns:
        True if successful
    """
    client = get_redis_client()
    if not client:
        return False
    
    try:
        if ttl is None:
            ttl = int(os.getenv("CACHE_TTL", "3600"))
        
        serialized = json.dumps(value)
        client.setex(key, ttl, serialized)
        return True
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        return False


def cache_delete(key: str) -> bool:
    """
    Delete key from cache.
    
    Args:
        key: Cache key
    
    Returns:
        True if successful
    """
    client = get_redis_client()
    if not client:
        return False
    
    try:
        client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Cache delete error: {e}")
        return False


def cache_clear_pattern(pattern: str) -> int:
    """
    Clear all keys matching pattern.
    
    Args:
        pattern: Redis pattern (e.g., "query:*")
    
    Returns:
        Number of keys deleted
    """
    client = get_redis_client()
    if not client:
        return 0
    
    try:
        keys = client.keys(pattern)
        if keys:
            return client.delete(*keys)
        return 0
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return 0


def cached(prefix: str, ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Usage:
        @cached("query_results", ttl=300)
        def search_query(query: str, top_k: int = 10):
            # expensive operation
            return results
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if cache is available
            if not is_cache_available():
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = cache_get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache HIT: {cache_key}")
                return cached_value
            
            # Cache miss - execute function
            logger.debug(f"Cache MISS: {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Convenience function for query caching
def cache_query_results(query: str, method: str, results: list, ttl: int = 300):
    """Cache query results."""
    key = generate_cache_key("query", query, method)
    return cache_set(key, results, ttl)


def get_cached_query_results(query: str, method: str) -> Optional[list]:
    """Get cached query results."""
    key = generate_cache_key("query", query, method)
    return cache_get(key)


# Stats
def get_cache_stats() -> dict:
    """Get cache statistics."""
    client = get_redis_client()
    if not client:
        return {"available": False}
    
    try:
        info = client.info("stats")
        return {
            "available": True,
            "total_keys": client.dbsize(),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {"available": False, "error": str(e)}
