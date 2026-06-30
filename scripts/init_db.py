"""
Initialize database schema and required tables.
Run this before first use to set up the database structure.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.database import get_db_connection, execute_query
from common.logging_config import setup_logging

logger = setup_logging("init_db")


def create_tables():
    """Create all required database tables."""
    
    # Documents table
    create_documents_table = """
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        doc_id VARCHAR(255) UNIQUE NOT NULL,
        title TEXT,
        text TEXT NOT NULL,
        dataset VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id);
    CREATE INDEX IF NOT EXISTS idx_dataset ON documents(dataset);
    """
    
    # Queries table
    create_queries_table = """
    CREATE TABLE IF NOT EXISTS queries (
        id SERIAL PRIMARY KEY,
        query_id VARCHAR(255) UNIQUE NOT NULL,
        text TEXT NOT NULL,
        dataset VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_query_id ON queries(query_id);
    CREATE INDEX IF NOT EXISTS idx_query_dataset ON queries(dataset);
    """
    
    # Query relevance (qrels) table
    create_qrels_table = """
    CREATE TABLE IF NOT EXISTS query_relevance (
        id SERIAL PRIMARY KEY,
        query_id VARCHAR(255) NOT NULL,
        doc_id VARCHAR(255) NOT NULL,
        relevance INTEGER NOT NULL,
        dataset VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(query_id, doc_id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_qrels_query ON query_relevance(query_id);
    CREATE INDEX IF NOT EXISTS idx_qrels_doc ON query_relevance(doc_id);
    CREATE INDEX IF NOT EXISTS idx_qrels_dataset ON query_relevance(dataset);
    """
    
    # Search history table
    create_search_history_table = """
    CREATE TABLE IF NOT EXISTS search_history (
        id SERIAL PRIMARY KEY,
        query TEXT NOT NULL,
        method VARCHAR(50),
        num_results INTEGER,
        execution_time_ms FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_search_created ON search_history(created_at DESC);
    """
    
    # Indices metadata table
    create_indices_meta_table = """
    CREATE TABLE IF NOT EXISTS indices_metadata (
        id SERIAL PRIMARY KEY,
        index_type VARCHAR(50) UNIQUE NOT NULL,
        dataset VARCHAR(100),
        num_documents INTEGER,
        build_time_seconds FLOAT,
        file_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_indices_type ON indices_metadata(index_type);
    """
    
    try:
        conn = get_db_connection()
        
        logger.info("Creating documents table...")
        execute_query(conn, create_documents_table)
        
        logger.info("Creating queries table...")
        execute_query(conn, create_queries_table)
        
        logger.info("Creating query_relevance table...")
        execute_query(conn, create_qrels_table)
        
        logger.info("Creating search_history table...")
        execute_query(conn, create_search_history_table)
        
        logger.info("Creating indices_metadata table...")
        execute_query(conn, create_indices_meta_table)
        
        conn.close()
        logger.info("✅ Database schema initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return False


def verify_tables():
    """Verify that all tables were created successfully."""
    check_query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
    ORDER BY table_name;
    """
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(check_query)
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        logger.info(f"Found tables: {', '.join(tables)}")
        
        expected_tables = ['documents', 'queries', 'query_relevance', 
                          'search_history', 'indices_metadata']
        
        missing = set(expected_tables) - set(tables)
        if missing:
            logger.warning(f"Missing tables: {', '.join(missing)}")
            return False
        else:
            logger.info("✅ All required tables exist!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to verify tables: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting database initialization...")
    
    if create_tables():
        if verify_tables():
            logger.info("🎉 Database is ready!")
            sys.exit(0)
        else:
            logger.error("❌ Table verification failed")
            sys.exit(1)
    else:
        logger.error("❌ Database initialization failed")
        sys.exit(1)
