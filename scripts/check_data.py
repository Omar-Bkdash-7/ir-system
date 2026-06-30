"""
Check the status of datasets, indices, and database.
Useful to see what data is available before building indices.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.database import get_db_connection
from common.logging_config import setup_logging

logger = setup_logging("check_data")


def check_database():
    """Check database status and document count."""
    logger.info("=" * 60)
    logger.info("DATABASE STATUS")
    logger.info("=" * 60)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check documents
        cursor.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cursor.fetchone()[0]
        logger.info(f"📄 Documents in database: {doc_count:,}")
        
        if doc_count > 0:
            cursor.execute("SELECT DISTINCT dataset FROM documents;")
            datasets = [row[0] for row in cursor.fetchall()]
            logger.info(f"   Datasets: {', '.join(datasets)}")
        
        # Check queries
        cursor.execute("SELECT COUNT(*) FROM queries;")
        query_count = cursor.fetchone()[0]
        logger.info(f"❓ Queries in database: {query_count:,}")
        
        # Check qrels
        cursor.execute("SELECT COUNT(*) FROM query_relevance;")
        qrel_count = cursor.fetchone()[0]
        logger.info(f"🔗 Query relevance judgments: {qrel_count:,}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")


def check_datasets():
    """Check downloaded datasets."""
    logger.info("\n" + "=" * 60)
    logger.info("DATASETS STATUS")
    logger.info("=" * 60)
    
    datasets_dir = Path("/app/data/datasets")
    
    if not datasets_dir.exists():
        logger.warning("⚠️  Datasets directory not found")
        return
    
    # Check .ir_datasets cache
    ir_datasets_dir = datasets_dir / ".ir_datasets"
    if ir_datasets_dir.exists():
        subdirs = [d for d in ir_datasets_dir.iterdir() if d.is_dir()]
        logger.info(f"📦 IR Datasets cached: {len(subdirs)} dataset(s)")
        for d in subdirs:
            size_mb = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / (1024 * 1024)
            logger.info(f"   - {d.name}: {size_mb:.2f} MB")
    else:
        logger.warning("⚠️  No IR datasets cache found")


def check_processed_data():
    """Check processed data."""
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSED DATA STATUS")
    logger.info("=" * 60)
    
    processed_dir = Path("/app/data/processed")
    
    if not processed_dir.exists():
        logger.warning("⚠️  Processed data directory not found")
        return
    
    files = list(processed_dir.glob("*.jsonl"))
    if files:
        logger.info(f"📝 Processed files: {len(files)}")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            lines = sum(1 for _ in open(f, 'r', encoding='utf-8'))
            logger.info(f"   - {f.name}: {lines:,} documents, {size_mb:.2f} MB")
    else:
        logger.warning("⚠️  No processed data files found")


def check_indices():
    """Check built indices."""
    logger.info("\n" + "=" * 60)
    logger.info("INDICES STATUS")
    logger.info("=" * 60)
    
    indices_dir = Path("/app/data/indices")
    
    if not indices_dir.exists():
        logger.warning("⚠️  Indices directory not found")
        return
    
    # Check for different index types
    index_files = {
        "TF-IDF": ["tfidf_matrix.npz", "tfidf_vectorizer.pkl", "tfidf_doc_ids.json"],
        "BM25": ["bm25_index.pkl", "doc_id_map.json"],
        "Embeddings": ["embeddings.npy", "doc_id_map.json"]
    }
    
    for index_type, files in index_files.items():
        found = all((indices_dir / f).exists() for f in files)
        if found:
            total_size = sum((indices_dir / f).stat().st_size for f in files)
            size_mb = total_size / (1024 * 1024)
            logger.info(f"✅ {index_type}: {size_mb:.2f} MB")
        else:
            logger.warning(f"❌ {index_type}: Not built")


def check_metadata():
    """Check metadata file."""
    logger.info("\n" + "=" * 60)
    logger.info("METADATA")
    logger.info("=" * 60)
    
    metadata_file = Path("/app/data/metadata.json")
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info("📋 Metadata file found:")
        for key, value in metadata.items():
            logger.info(f"   {key}: {value}")
    else:
        logger.warning("⚠️  No metadata file found")


if __name__ == "__main__":
    logger.info("🔍 Checking data status...\n")
    
    check_database()
    check_datasets()
    check_processed_data()
    check_indices()
    check_metadata()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Data check completed!")
    logger.info("=" * 60)
