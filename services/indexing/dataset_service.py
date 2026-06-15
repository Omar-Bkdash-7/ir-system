import ir_datasets
import json
from pathlib import Path
from datetime import datetime
import time
from common.config import get_config

class DatasetValidationError(Exception):
    pass

class ForbiddenDatasetError(Exception):
    pass

def validate_and_store_metadata(dataset_id: str = None) -> dict:
    if dataset_id is None:
        dataset_id = get_config("dataset.ir_datasets_id", "beir/quora")
    
    max_retries = get_config("dataset.validation_retries", 3)
    retry_delay = get_config("dataset.validation_retry_delay", 10)
    min_docs = get_config("dataset.min_docs_threshold", 200000)
    min_queries = get_config("dataset.min_queries_threshold", 1000)
    min_qrels = get_config("dataset.min_qrels_threshold", 1000)
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            docs_ds = ir_datasets.load(dataset_id)
            test_ds = ir_datasets.load(f"{dataset_id}/test")
            
            doc_count = docs_ds.docs_count()
            query_count = test_ds.queries_count()
            qrels_count = test_ds.qrels_count()
            
            if doc_count < min_docs:
                raise DatasetValidationError(
                    f"Document count {doc_count} is below minimum threshold of {min_docs}"
                )
            
            if query_count < min_queries:
                raise DatasetValidationError(
                    f"Query count {query_count} is below minimum threshold of {min_queries}"
                )
            
            if qrels_count < min_qrels:
                raise DatasetValidationError(
                    f"Qrels count {qrels_count} is below minimum threshold of {min_qrels}"
                )
            
            metadata = {
                "dataset_name": dataset_id,
                "doc_count": doc_count,
                "query_count": query_count,
                "qrels_count": qrels_count,
                "timestamp": datetime.utcnow().isoformat(),
                "validation_status": "success"
            }
            
            metadata_path = Path(get_config("paths.metadata", "data/metadata.json"))
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise DatasetValidationError(
                    f"Failed to load dataset after {max_retries} retries: {str(e)}"
                )
            time.sleep(retry_delay)

def get_metadata() -> dict:
    metadata_path = Path(get_config("paths.metadata", "data/metadata.json"))
    if not metadata_path.exists():
        raise FileNotFoundError("Metadata file not found")
    
    with open(metadata_path) as f:
        return json.load(f)
