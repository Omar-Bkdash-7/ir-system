from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ContextDocument(BaseModel):
    doc_id: str = Field(description="Document ID")
    text: str = Field(description="Document text content")
    score: float = Field(description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")


class GenerationConfig(BaseModel):
    model: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Maximum tokens to generate")
    use_context: bool = Field(default=True, description="Whether to use retrieved context")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")


class RAGRequest(BaseModel):
    query: str = Field(description="User query for RAG")
    dataset: str = Field(description="Dataset to retrieve from (e.g., 'cranfield', 'vaswani')")
    retrieval_model: str = Field(default="embedding", description="Retrieval model: tfidf, embedding, bm25, hybrid")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    bm25_k1: Optional[float] = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    context_documents: List[ContextDocument] = Field(description="Retrieved context documents")
    model_used: str = Field(description="LLM model used for generation")
    retrieval_model: str = Field(description="Retrieval model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
