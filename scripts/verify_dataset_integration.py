"""
Dataset Integration Verification Script

Tests the multi-dataset architecture to ensure all components work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging_config import setup_logging

logger = setup_logging("verify_integration")


def test_config_helpers():
    """Test config helper functions"""
    logger.info("=" * 70)
    logger.info("TEST 1: Config Helper Functions")
    logger.info("=" * 70)
    
    try:
        from common.config import get_dataset_specific_path
        
        # Test dataset slug generation
        test_cases = [
            ("beir/quora", "beir_quora"),
            ("cranfield", "cranfield"),
            ("beir/nfcorpus", "beir_nfcorpus"),
            ("beir-scifact", "beir_scifact"),  # already has dash
        ]
        
        for dataset_name, expected_slug in test_cases:
            path = get_dataset_specific_path(
                "indices.bm25_index",
                dataset_name,
                "data/indices/{dataset}/bm25_index.pkl"
            )
            
            if expected_slug in path:
                logger.info(f"✅ {dataset_name:20s} → {expected_slug:20s} ✓")
            else:
                logger.error(f"❌ {dataset_name:20s} → Expected {expected_slug} in path")
                return False
        
        logger.info("✅ Config helpers working correctly\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config helper test failed: {e}")
        return False


def test_metadata_system():
    """Test metadata management"""
    logger.info("=" * 70)
    logger.info("TEST 2: Metadata Management System")
    logger.info("=" * 70)
    
    try:
        from services.indexing.dataset.metadata_manager import DatasetMetadataManager
        
        manager = DatasetMetadataManager()
        
        # Check if metadata directory exists
        metadata_dir = manager.metadata_dir
        logger.info(f"Metadata directory: {metadata_dir}")
        
        # List existing datasets
        datasets = manager.list_datasets()
        logger.info(f"Found {len(datasets)} datasets with metadata:")
        
        for ds in datasets:
            logger.info(f"  - {ds.get('dataset_name', 'Unknown')}: "
                       f"{ds.get('doc_count', 0)} docs, "
                       f"{ds.get('query_count', 0)} queries")
        
        # Check active dataset
        active = manager.get_active_dataset()
        if active:
            logger.info(f"✅ Active dataset: {active}")
        else:
            logger.warning("⚠️  No active dataset set")
        
        logger.info("✅ Metadata system working correctly\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Metadata test failed: {e}")
        return False


def test_database_structure():
    """Test database table structure"""
    logger.info("=" * 70)
    logger.info("TEST 3: Database Table Structure")
    logger.info("=" * 70)
    
    try:
        from common.database import get_db_connection
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # List all dataset tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name LIKE 'dataset_%'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        logger.info(f"Found {len(tables)} dataset tables:")
        
        for (table_name,) in tables:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            # Extract dataset name
            dataset_name = table_name.replace("dataset_", "").replace("_", "/")
            
            logger.info(f"  - {table_name:30s} → {dataset_name:20s} ({count:,} docs)")
        
        cursor.close()
        conn.close()
        
        if len(tables) > 0:
            logger.info("✅ Database structure correct\n")
            return True
        else:
            logger.warning("⚠️  No dataset tables found (may need to upload datasets)\n")
            return True  # Not an error, just empty
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        logger.info("Note: Database might not be running or accessible\n")
        return False


def test_indexer_paths():
    """Test that indexers use correct paths"""
    logger.info("=" * 70)
    logger.info("TEST 4: Indexer Path Resolution")
    logger.info("=" * 70)
    
    try:
        from services.indexing.indexers import TFIDFIndexer, BM25Indexer, EmbeddingIndexer
        
        test_dataset = "beir/quora"
        
        # Test TF-IDF Indexer
        tfidf = TFIDFIndexer(dataset_name=test_dataset)
        logger.info(f"TF-IDF paths for '{test_dataset}':")
        logger.info(f"  - Vectorizer: {tfidf.vectorizer_path}")
        logger.info(f"  - Matrix:     {tfidf.matrix_path}")
        logger.info(f"  - Doc IDs:    {tfidf.doc_ids_path}")
        
        # Verify slug in path
        if "beir_quora" in str(tfidf.vectorizer_path):
            logger.info("  ✅ Dataset slug correct")
        else:
            logger.error("  ❌ Dataset slug missing or incorrect")
            return False
        
        # Test BM25 Indexer
        bm25 = BM25Indexer(dataset_name=test_dataset)
        logger.info(f"\nBM25 paths for '{test_dataset}':")
        logger.info(f"  - Index: {bm25.index_path}")
        
        if "beir_quora" in str(bm25.index_path):
            logger.info("  ✅ Dataset slug correct")
        else:
            logger.error("  ❌ Dataset slug missing or incorrect")
            return False
        
        # Test Embedding Indexer
        embedding = EmbeddingIndexer(dataset_name=test_dataset)
        logger.info(f"\nEmbedding paths for '{test_dataset}':")
        logger.info(f"  - Embeddings: {embedding.embeddings_path}")
        logger.info(f"  - Doc Map:    {embedding.doc_id_map_path}")
        
        if "beir_quora" in str(embedding.embeddings_path):
            logger.info("  ✅ Dataset slug correct")
        else:
            logger.error("  ❌ Dataset slug missing or incorrect")
            return False
        
        logger.info("\n✅ All indexers use correct paths\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Indexer path test failed: {e}")
        return False


def test_retriever_integration():
    """Test that retrievers can load active dataset"""
    logger.info("=" * 70)
    logger.info("TEST 5: Retriever Integration")
    logger.info("=" * 70)
    
    try:
        from common.config import get_active_dataset_path
        
        # Test that helper function works
        try:
            bm25_path = get_active_dataset_path("indices.bm25_index")
            logger.info(f"BM25 path for active dataset: {bm25_path}")
            
            tfidf_path = get_active_dataset_path("indices.tfidf_vectorizer")
            logger.info(f"TF-IDF path for active dataset: {tfidf_path}")
            
            embedding_path = get_active_dataset_path("indices.embeddings")
            logger.info(f"Embedding path for active dataset: {embedding_path}")
            
            logger.info("✅ Retriever path resolution working\n")
            return True
            
        except ValueError as e:
            logger.warning(f"⚠️  No active dataset: {e}")
            logger.info("This is expected if no dataset has been validated yet\n")
            return True  # Not an error
        
    except Exception as e:
        logger.error(f"❌ Retriever test failed: {e}")
        return False


def check_file_structure():
    """Check that expected directories exist"""
    logger.info("=" * 70)
    logger.info("TEST 6: File Structure")
    logger.info("=" * 70)
    
    project_root = Path(__file__).parent.parent
    
    expected_dirs = [
        "data",
        "data/datasets",
        "data/indices",
        "data/metadata",
        "data/cache",
    ]
    
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            logger.info(f"✅ {dir_path:30s} exists")
        else:
            logger.warning(f"⚠️  {dir_path:30s} missing (will be created on first use)")
    
    # Check for indices subdirectories (dataset-specific)
    indices_dir = project_root / "data" / "indices"
    if indices_dir.exists():
        dataset_dirs = [d for d in indices_dir.iterdir() if d.is_dir()]
        if dataset_dirs:
            logger.info(f"\nFound {len(dataset_dirs)} dataset index directories:")
            for dataset_dir in dataset_dirs:
                logger.info(f"  - {dataset_dir.name}")
    
    logger.info("\n✅ File structure check complete\n")
    return True


def main():
    """Run all verification tests"""
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 15 + "DATASET INTEGRATION VERIFICATION" + " " * 21 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    logger.info("")
    
    tests = [
        ("Config Helpers", test_config_helpers),
        ("Metadata System", test_metadata_system),
        ("Database Structure", test_database_structure),
        ("Indexer Paths", test_indexer_paths),
        ("Retriever Integration", test_retriever_integration),
        ("File Structure", check_file_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
