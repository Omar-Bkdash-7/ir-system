import jsonlines
from pathlib import Path
from common.config import get_config
from common.database import init_db, insert_documents
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_documents_to_db():
    raw_path = Path(get_config("paths.raw_docs", "data/raw/quora_docs.jsonl"))
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw documents not found at {raw_path}")
    
    logger.info("Initializing database...")
    init_db()
    
    logger.info(f"Loading documents from {raw_path}")
    
    with jsonlines.open(raw_path) as reader:
        docs = list(reader)
    
    logger.info(f"Total documents to insert: {len(docs)}")
    
    batch_size = 10000
    for i in tqdm(range(0, len(docs), batch_size), desc="Inserting batches"):
        batch = docs[i:i + batch_size]
        insert_documents(batch)
    
    logger.info("All documents loaded successfully")

if __name__ == "__main__":
    load_documents_to_db()
