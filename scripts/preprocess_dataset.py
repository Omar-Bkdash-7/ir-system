import jsonlines
from pathlib import Path
from services.preprocessing.preprocessing_service import preprocess
from common.config import get_config
from common.logging_config import setup_logging
from tqdm import tqdm

logger = setup_logging("preprocess_dataset")


def preprocess_all_documents():
    raw_path = Path(get_config("paths.raw_docs", "data/raw/quora_docs.jsonl"))
    processed_path = Path(get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl"))

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw documents not found at {raw_path}")

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting preprocessing: {raw_path} -> {processed_path}")

    with jsonlines.open(raw_path) as reader, \
         jsonlines.open(processed_path, mode="w") as writer:

        docs = list(reader)
        logger.info(f"Total documents to process: {len(docs)}")

        processed_count = 0
        empty_count = 0

        for doc in tqdm(docs, desc="Preprocessing documents"):
            tokens = preprocess(doc["text"], mode="document")

            if not tokens:
                empty_count += 1

            writer.write({
                "doc_id": doc["doc_id"],
                "tokens": tokens,
                "text": doc["text"]
            })
            processed_count += 1

        logger.info(f"Preprocessing complete: {processed_count} documents processed")
        logger.info(f"Empty token results: {empty_count}")
        logger.info(f"Output: {processed_path}")


if __name__ == "__main__":
    preprocess_all_documents()
