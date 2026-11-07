"""Framework-free core of the retrieval service (numpy/scipy only: no FastAPI, no pydantic).

kernels   pure scoring functions            compact   memory-mappable cache of the BM25 / TF-IDF indices
indexes   Bm25Index / TfidfIndex / DenseIndex       store     lazy loading, invalidation, memory budget
query     one query-preparation policy      encoder   query embeddings (lazy model, LRU)
texts     document texts from PostgreSQL    service   retrieve / hybrid serial / hybrid parallel / status
"""
