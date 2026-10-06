import os
import httpx
from typing import List
from services.rag.schemas import RAGRequest, RAGResponse, ContextDocument
from services.rag.schemas.rag_schemas import GenerationConfig
from services.rag.generators import HuggingFaceGenerator
from common.logging_config import setup_logging
from common.exceptions import ServiceError

logger = setup_logging("rag.service")


class RAGService:

    def __init__(self):
        self.retrieval_url = os.getenv("RETRIEVAL_URL", "http://retrieval:8003")
        self.timeout = 120.0
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        if not hf_token or not hf_token.strip():
            logger.warning(
                "HUGGINGFACE_API_TOKEN is not set. "
                "RAG generation will fail. Set the token in your .env file."
            )
        logger.info("RAGService initialized (HuggingFace remote only)")

    async def retrieve_context(self, request: RAGRequest) -> List[ContextDocument]:
        url = f"{self.retrieval_url}/search"

        payload = {
            "query": request.query,
            "dataset": request.dataset,
            "model": request.retrieval_model,
            "top_k": request.top_k
        }

        if request.retrieval_model == "bm25":
            payload["bm25_k1"] = request.bm25_k1
            payload["bm25_b"] = request.bm25_b

        logger.info(f"Retrieving context from {url} with model {request.retrieval_model}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

            documents = []
            for result in data.get("results", [])[:request.top_k]:
                doc = ContextDocument(
                    doc_id=result.get("doc_id", ""),
                    text=result.get("text", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {})
                )
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} documents")
            return documents

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error retrieving context: {e.response.status_code} - {e.response.text}")
            raise ServiceError(f"Retrieval service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            raise ServiceError(f"Failed to retrieve context: {str(e)}")

    async def generate_answer(self, request: RAGRequest) -> RAGResponse:
        logger.info(f"Processing RAG request for query: '{request.query[:50]}...'")

        context_docs = await self.retrieve_context(request)

        if not context_docs:
            logger.warning("No context documents retrieved")
            return RAGResponse(
                query=request.query,
                answer="No relevant documents found to answer this question.",
                context_documents=[],
                model_used="none",
                retrieval_model=request.retrieval_model,
                metadata={"status": "no_context"}
            )

        try:
            generator = HuggingFaceGenerator(request.generation_config)
            answer = generator.generate(request.query, context_docs)
            actual_model_used = f"{request.generation_config.model} (remote)"
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise ServiceError(str(e))
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            raise ServiceError(f"Failed to generate answer: {str(e)}")

        logger.info("RAG answer generated successfully")

        return RAGResponse(
            query=request.query,
            answer=answer,
            context_documents=context_docs,
            model_used=actual_model_used,
            retrieval_model=request.retrieval_model,
            metadata={
                "status": "success",
                "num_docs_retrieved": len(context_docs),
                "temperature": request.generation_config.temperature,
                "max_tokens": request.generation_config.max_tokens
            }
        )
