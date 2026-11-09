"""The RAG service must speak the retrieval service's real contract.

/retrieve only accepts tfidf/embedding/bm25 and answers 422 for "hybrid", which the RAG schema and /models advertised:
the retrieval service knows hybrid searching only as a fusion of models running in parallel.
"""
import asyncio
import json
import unittest

import httpx

from services.rag.config import RAGSettings
from services.rag.rag_service import RAGService
from services.rag.schemas import RAGRequest


class RetrievalContractTest(unittest.TestCase):
    def _retrieve(self, retrieval_model):
        calls = []

        async def handler(request):
            await request.aread()
            calls.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, json={"results": [{"doc_id": "d1", "text": "text", "score": 0.5}]})

        async def run():
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                service = RAGService(RAGSettings(), None, client)
                return await service.retrieve_context(
                    RAGRequest(query="who are you", dataset="cranfield", retrieval_model=retrieval_model))
            finally:
                await client.aclose()

        return calls, asyncio.run(run())

    def test_hybrid_uses_the_parallel_endpoint_with_models_the_service_accepts(self):
        calls, documents = self._retrieve("hybrid")
        url, payload = calls[0]
        self.assertTrue(url.endswith("/retrieve/hybrid/parallel"))
        self.assertEqual(payload["models"], ["tfidf", "bm25"])
        self.assertEqual(payload["fusion_method"], "rrf")
        self.assertGreaterEqual(payload["candidates_per_model"], 10)
        self.assertEqual([document.doc_id for document in documents], ["d1"])

    def test_a_single_model_keeps_the_plain_endpoint(self):
        calls, _ = self._retrieve("embedding")
        url, payload = calls[0]
        self.assertTrue(url.endswith("/retrieve"))
        self.assertEqual(payload["model"], "embedding")
        self.assertNotIn("models", payload)


if __name__ == "__main__":
    unittest.main()
