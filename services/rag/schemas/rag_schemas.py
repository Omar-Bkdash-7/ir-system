from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ContextDocument(BaseModel):
    doc_id: str = Field(description="Document ID")
    text: str = Field(description="Document text content")
    score: float = Field(description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")


class GenerationConfig(BaseModel):
    # Old clients may still send "model" (e.g. "flan-t5-large", "gpt-3.5-turbo"). It is accepted
    # and IGNORED: the remote model is chosen by the server (RAG_REMOTE_MODEL).
    model: Optional[str] = Field(default=None, description="Deprecated, ignored.")
    mode: Optional[Literal["auto", "remote", "local"]] = Field(
        default=None, description="auto = remote then local fallback. None = server default."
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Maximum tokens to generate")
    use_context: bool = Field(default=True, description="Whether to use retrieved context")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")


class RAGRequest(BaseModel):
    query: str = Field(description="User query for RAG")
    dataset: str = Field(description="Dataset to retrieve from (e.g., 'cranfield', 'vaswani')")
    retrieval_model: str = Field(default="embedding", description="tfidf, embedding, bm25, hybrid")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    bm25_k1: Optional[float] = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    # keep the existing field names; silence pydantic's "model_" protected-namespace warning
    model_config = ConfigDict(protected_namespaces=())

    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer ('' when generation failed)")
    context_documents: List[ContextDocument] = Field(description="Retrieved context documents")
    model_used: str = Field(description="e.g. 'meta-llama/Llama-3.1-8B-Instruct (remote)'")
    retrieval_model: str = Field(description="Retrieval model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="status, source, fallback info, latency...")
