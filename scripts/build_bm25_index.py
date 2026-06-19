import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.indexing.bm25_indexer import BM25Indexer
from common.logging_config import setup_logging

logger = setup_logging("build_bm25_index")

def main():
    logger.info("Starting BM25 index build")
    
    indexer = BM25Indexer()
    
    if indexer.exists():
        logger.warning("BM25 index already exists. Delete it or use force_rebuild=True")
        response = input("BM25 index exists. Rebuild? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return
    
    try:
        logger.info("Building BM25 index...")
        indexer.build()
        
        logger.info("Saving BM25 index...")
        indexer.save()
        
        logger.info("BM25 index build complete!")
        logger.info(f"Total documents: {indexer.total_docs}")
        logger.info(f"Vocabulary size: {len(indexer.inverted_index)}")
        logger.info(f"Average document length: {indexer.avg_doc_length:.2f}")
        
    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
