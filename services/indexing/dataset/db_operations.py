"""
Database operations for dataset management.
"""

import logging
from contextlib import contextmanager
from typing import Dict, Any, Tuple, List, Iterator

import ir_datasets
from psycopg2.extras import execute_batch
from psycopg2.extensions import connection as PgConnection, cursor as PgCursor

from common.config import get_config
from common.database import get_db_connection
from services.indexing.dataset.constants import DEFAULT_BATCH_SIZE, DB_BATCH_PAGE_SIZE


logger = logging.getLogger(__name__)


@contextmanager
def database_cursor(conn: PgConnection) -> Iterator[PgCursor]:
    """
    Context manager for safe database cursor handling.
    
    Ensures cursor is properly closed even if exceptions occur.
    
    Args:
        conn: PostgreSQL connection object
        
    Yields:
        Database cursor
    """
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


class DatasetUploader:
    """
    Uploads dataset documents to PostgreSQL database.
    
    Implements efficient batch insertion with transaction management.
    Creates a separate table for each dataset for better isolation and performance.
    """
    
    def __init__(self):
        """Initialize uploader with configuration."""
        self.base_table_prefix = get_config("database.table_prefix", "dataset")
    
    def _get_table_name(self, dataset_id: str) -> str:
        """
        Generate safe table name from dataset ID.
        
        Args:
            dataset_id: Dataset identifier (e.g., 'beir/quora')
            
        Returns:
            Safe PostgreSQL table name (e.g., 'dataset_beir_quora')
        """
        # Replace special characters with underscores
        safe_name = dataset_id.replace('/', '_').replace('-', '_').replace('.', '_')
        return f"{self.base_table_prefix}_{safe_name}"
    
    def upload_to_db(
        self, 
        dataset_id: str, 
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> Dict[str, Any]:
        """
        Upload dataset documents to PostgreSQL in a dedicated table.
        
        Creates a new table for the dataset if it doesn't exist.
        If table exists, drops and recreates it to ensure clean upload.
        
        Args:
            dataset_id: IR datasets identifier
            batch_size: Number of documents per batch insert
            
        Returns:
            Dictionary with upload statistics (inserted, skipped counts)
            
        Raises:
            ValueError: If dataset has no documents iterator
            psycopg2.Error: If database operation fails
        """
        logger.info(f"Starting upload of dataset '{dataset_id}' to PostgreSQL")
        
        table_name = self._get_table_name(dataset_id)
        logger.info(f"Target table: {table_name}")
        
        ds = ir_datasets.load(dataset_id)
        
        if not hasattr(ds, "docs_iter"):
            error_msg = f"Dataset '{dataset_id}' has no docs_iter() method"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        conn = get_db_connection()
        try:
            # Check if table exists and ask for confirmation via warning
            table_exists = self._table_exists(conn, table_name)
            
            if table_exists:
                logger.warning(f"Table '{table_name}' already exists. Dropping and recreating...")
                self._drop_table(conn, table_name)
            
            self._create_table(conn, table_name)
            
            inserted, skipped = self._process_documents(
                conn,
                table_name,
                ds.docs_iter(), 
                batch_size
            )
            
            logger.info(
                f"Upload complete for '{dataset_id}': "
                f"inserted={inserted}, skipped={skipped}"
            )
            
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "table": table_name,
                "inserted": inserted,
                "skipped": skipped
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"DB upload failed for '{dataset_id}': {e}", exc_info=True)
            raise
        finally:
            conn.close()
    
    def _table_exists(self, conn: PgConnection, table_name: str) -> bool:
        """
        Check if table exists in database.
        
        Args:
            conn: Database connection
            table_name: Table name to check
            
        Returns:
            True if table exists, False otherwise
        """
        with database_cursor(conn) as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name = %s
                )
            """, (table_name,))
            return cur.fetchone()[0]
    
    def _drop_table(self, conn: PgConnection, table_name: str) -> None:
        """
        Drop table if it exists.
        
        Args:
            conn: Database connection
            table_name: Table name to drop
        """
        with database_cursor(conn) as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            conn.commit()
            logger.info(f"Dropped table '{table_name}'")
    
    def _create_table(self, conn: PgConnection, table_name: str) -> None:
        """
        Create documents table for specific dataset.
        
        Args:
            conn: Database connection
            table_name: Table name to create
        """
        with database_cursor(conn) as cur:
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    doc_id VARCHAR(255) PRIMARY KEY,
                    text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for better query performance
            cur.execute(f"""
                CREATE INDEX idx_{table_name}_doc_id 
                ON {table_name}(doc_id)
            """)
            
            # Add full-text search index
            cur.execute(f"""
                CREATE INDEX idx_{table_name}_text_gin 
                ON {table_name} USING gin(to_tsvector('english', text))
            """)
            
            conn.commit()
            logger.info(f"Created table '{table_name}' with indexes")
    
    def _process_documents(
        self, 
        conn: PgConnection,
        table_name: str,
        docs_iter, 
        batch_size: int
    ) -> Tuple[int, int]:
        """
        Process and insert documents in batches.
        
        Args:
            conn: Database connection
            table_name: Target table name
            docs_iter: Iterator yielding document objects
            batch_size: Number of documents per batch
            
        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        with database_cursor(conn) as cur:
            inserted = 0
            skipped = 0
            batch: List[Tuple[str, str]] = []
            
            for idx, doc in enumerate(docs_iter, 1):
                text = self._extract_text(doc)
                
                if not text or not text.strip():
                    skipped += 1
                    continue
                
                batch.append((doc.doc_id, text))
                
                if len(batch) >= batch_size:
                    inserted += self._flush_batch(cur, conn, table_name, batch)
                    batch = []
                    
                    # Progress logging every 10k documents
                    if idx % 10000 == 0:
                        logger.info(f"Processed {idx} documents...")
            
            # Flush remaining documents
            if batch:
                inserted += self._flush_batch(cur, conn, table_name, batch)
            
            logger.info(f"Total processed: {inserted + skipped} documents")
            
            return inserted, skipped
    
    @staticmethod
    def _extract_text(doc) -> str:
        """
        Extract text content from document object.
        
        Tries multiple common text field names used in ir_datasets.
        
        Args:
            doc: Document object from ir_datasets
            
        Returns:
            Extracted text content or empty string
        """
        return (
            getattr(doc, "text", None) or 
            getattr(doc, "body", None) or 
            getattr(doc, "content", None) or
            ""
        )
    
    def _flush_batch(
        self, 
        cur: PgCursor, 
        conn: PgConnection,
        table_name: str,
        batch: List[Tuple[str, str]]
    ) -> int:
        """
        Insert a batch of documents into database.
        
        Uses execute_batch for efficient bulk insertion.
        
        Args:
            cur: Database cursor
            conn: Database connection
            table_name: Target table name
            batch: List of (doc_id, text) tuples
            
        Returns:
            Number of documents in batch
        """
        execute_batch(
            cur,
            f"""
            INSERT INTO {table_name} (doc_id, text) 
            VALUES (%s, %s) 
            ON CONFLICT (doc_id) DO UPDATE SET
                text = EXCLUDED.text,
                created_at = CURRENT_TIMESTAMP
            """,
            batch,
            page_size=DB_BATCH_PAGE_SIZE,
        )
        conn.commit()
        logger.debug(f"Flushed batch of {len(batch)} documents to {table_name}")
        
        return len(batch)


class DatabaseInfoRetriever:
    """
    Retrieves information about database state.
    
    Provides database statistics for all dataset tables.
    """
    
    def __init__(self):
        """Initialize retriever with configuration."""
        self.table_prefix = get_config("database.table_prefix", "dataset")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get database table information for all datasets.
        
        Lists all dataset tables and their statistics.
        
        Returns:
            Dictionary with all dataset tables and their info
        """
        conn = get_db_connection()
        try:
            with database_cursor(conn) as cur:
                # Get all dataset tables
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name LIKE %s
                    ORDER BY table_name
                """, (f"{self.table_prefix}_%",))
                
                tables = [row[0] for row in cur.fetchall()]
                
                datasets = []
                total_rows = 0
                
                for table in tables:
                    # Extract dataset_id from table name
                    dataset_id = table.replace(f"{self.table_prefix}_", "").replace("_", "/")
                    
                    # Get row count for this table
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    total_rows += count
                    
                    # Get table size
                    cur.execute(f"""
                        SELECT pg_size_pretty(pg_total_relation_size('{table}'))
                    """)
                    size = cur.fetchone()[0]
                    
                    datasets.append({
                        "dataset_id": dataset_id,
                        "table_name": table,
                        "doc_count": count,
                        "size": size
                    })
            
            result = {
                "total_tables": len(tables),
                "total_documents": total_rows,
                "datasets": datasets,
                "status": "healthy"
            }
            
            logger.debug(f"Database info retrieved: {len(datasets)} datasets")
            return result
            
        except Exception as e:
            logger.error(f"DB info retrieval failed: {e}", exc_info=True)
            return {
                "total_tables": 0,
                "total_documents": 0,
                "datasets": [],
                "status": "error",
                "error": str(e)
            }
        finally:
            conn.close()



class DatasetDatabaseDeleter:
    """
    Deletes dataset tables from PostgreSQL database.
    
    Provides safe deletion of specific dataset tables.
    """
    
    def __init__(self):
        """Initialize deleter with configuration."""
        self.table_prefix = get_config("database.table_prefix", "dataset")
        
        # Import here to avoid circular dependency
        from services.indexing.dataset.metadata_manager import DatasetMetadataManager
        self.metadata_manager = DatasetMetadataManager()
    
    def _get_table_name(self, dataset_id: str) -> str:
        """
        Generate safe table name from dataset ID.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Table name
        """
        safe_name = dataset_id.replace('/', '_').replace('-', '_').replace('.', '_')
        return f"{self.table_prefix}_{safe_name}"
    
    def delete_dataset_from_db(self, dataset_id: str) -> Dict[str, Any]:
        """
        Delete dataset table from PostgreSQL and clean up metadata.
        
        Drops the entire table for the specified dataset.
        Also cleans up metadata if dataset is deleted completely.
        
        Args:
            dataset_id: Dataset identifier to delete
            
        Returns:
            Dictionary with deletion statistics
            
        Raises:
            psycopg2.Error: If database operation fails
        """
        logger.info(f"Deleting dataset '{dataset_id}' from PostgreSQL")
        
        table_name = self._get_table_name(dataset_id)
        
        conn = get_db_connection()
        try:
            with database_cursor(conn) as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                """, (table_name,))
                
                exists = cur.fetchone()[0]
                
                if not exists:
                    logger.warning(f"Table '{table_name}' not found in database")
                    return {
                        "status": "not_found",
                        "dataset_id": dataset_id,
                        "table": table_name,
                        "message": f"Dataset '{dataset_id}' not found in database"
                    }
                
                # Get count before deletion
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
                
                # Drop the table
                cur.execute(f"DROP TABLE {table_name} CASCADE")
                conn.commit()
                
                logger.info(f"Dropped table '{table_name}' with {count} documents")
                
                # Clean up metadata (optional - only if local dataset is also deleted)
                # Check if local dataset still exists
                from services.indexing.dataset.path_resolver import DatasetPathResolver
                path_resolver = DatasetPathResolver()
                ir_home = path_resolver.get_ir_home()
                
                # Build path to dataset
                parts = dataset_id.strip("/").split("/")
                dataset_path = ir_home
                for part in parts:
                    dataset_path = dataset_path / part
                
                # If local dataset doesn't exist, clean up metadata too
                if not dataset_path.exists():
                    try:
                        self.metadata_manager.delete_dataset_metadata(dataset_id)
                        logger.info(f"Cleaned up metadata for '{dataset_id}' (local dataset not found)")
                    except Exception as e:
                        logger.warning(f"Failed to clean up metadata: {e}")
                
                return {
                    "status": "success",
                    "dataset_id": dataset_id,
                    "table": table_name,
                    "deleted": count,
                    "message": f"Successfully deleted table with {count} documents"
                }
                
        except Exception as e:
            conn.rollback()
            logger.error(f"DB deletion failed for '{dataset_id}': {e}", exc_info=True)
            raise
        finally:
            conn.close()
