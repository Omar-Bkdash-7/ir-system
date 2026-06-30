"""
Clear Redis cache - useful for testing or after index rebuilds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.cache import get_redis_client, cache_clear_pattern, get_cache_stats
from common.logging_config import setup_logging

logger = setup_logging("clear_cache")


def clear_all_cache():
    """Clear all cache."""
    logger.info("Clearing all cache...")
    
    client = get_redis_client()
    if not client:
        logger.error("Redis not available")
        return False
    
    try:
        # Clear everything
        count = cache_clear_pattern("*")
        logger.info(f"✅ Cleared {count} cache entries")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clear cache: {e}")
        return False


def clear_pattern(pattern: str):
    """Clear cache by pattern."""
    logger.info(f"Clearing cache pattern: {pattern}")
    
    count = cache_clear_pattern(pattern)
    logger.info(f"✅ Cleared {count} cache entries")


def show_stats():
    """Show cache statistics."""
    logger.info("Cache Statistics:")
    logger.info("=" * 60)
    
    stats = get_cache_stats()
    
    if not stats.get("available"):
        logger.warning("❌ Cache not available")
        return
    
    logger.info(f"Total keys: {stats['total_keys']:,}")
    logger.info(f"Cache hits: {stats['hits']:,}")
    logger.info(f"Cache misses: {stats['misses']:,}")
    logger.info(f"Hit rate: {stats['hit_rate']:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear Redis cache")
    parser.add_argument("--pattern", help="Clear specific pattern (e.g., 'query:*')")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--all", action="store_true", help="Clear all cache")
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
    elif args.pattern:
        clear_pattern(args.pattern)
    elif args.all:
        clear_all_cache()
    else:
        # Default: show stats
        show_stats()
