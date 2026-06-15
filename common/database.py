import psycopg2
from psycopg2.extras import execute_batch
from common.config import get_config
import logging

logger = logging.getLogger(__name__)

def get_connection():
    return psycopg2.connect(
        host=get_config("database.host", "postgres"),
        port=get_config("database.port", 5432),
        user=get_config("database.user", "ir_user"),
        password=get_config("database.password", "ir_pass"),
        database=get_config("database.database", "ir_db")
    )

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    
    table_name = get_config("database.table", "documents")
    
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            doc_id VARCHAR(255) PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)
    
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_doc_id ON {table_name}(doc_id)")
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Database table '{table_name}' initialized")

def insert_documents(documents: list[dict]):
    conn = get_connection()
    cur = conn.cursor()
    
    table_name = get_config("database.table", "documents")
    
    data = [(doc["doc_id"], doc["text"]) for doc in documents]
    
    execute_batch(
        cur,
        f"INSERT INTO {table_name} (doc_id, text) VALUES (%s, %s) ON CONFLICT (doc_id) DO NOTHING",
        data,
        page_size=1000
    )
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Inserted {len(documents)} documents into database")

def get_documents_by_ids(doc_ids: list[str]) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    
    table_name = get_config("database.table", "documents")
    
    cur.execute(
        f"SELECT doc_id, text FROM {table_name} WHERE doc_id = ANY(%s)",
        (doc_ids,)
    )
    
    results = {row[0]: row[1] for row in cur.fetchall()}
    
    cur.close()
    conn.close()
    
    return results
