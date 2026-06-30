"""
Refresh materialized views for better query performance.
Run this periodically (e.g., daily) or after significant data changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.database import get_db_connection, execute_query
from common.logging_config import setup_logging

logger = setup_logging("refresh_views")


def refresh_materialized_views():
    """Refresh all materialized views."""
    logger.info("Refreshing materialized views...")
    
    views = [
        "mv_recent_searches",
        "mv_popular_documents"
    ]
    
    try:
        conn = get_db_connection()
        
        for view in views:
            logger.info(f"  Refreshing {view}...")
            execute_query(conn, f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view};")
            logger.info(f"  ✅ {view} refreshed")
        
        conn.close()
        logger.info("✅ All materialized views refreshed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to refresh views: {e}")
        return False


if __name__ == "__main__":
    refresh_materialized_views()
