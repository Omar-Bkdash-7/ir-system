from abc import ABC, abstractmethod
from typing import List
from services.rag.schemas import ContextDocument, GenerationConfig


class BaseGenerator(ABC):
    
    def __init__(self, config: GenerationConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, query: str, context_docs: List[ContextDocument]) -> str:
        pass
    
    def _format_context(self, context_docs: List[ContextDocument]) -> str:
        if not context_docs:
            return ""
        
        formatted_parts = []
        for idx, doc in enumerate(context_docs, 1):
            formatted_parts.append(
                f"Document {idx} (ID: {doc.doc_id}, Score: {doc.score:.4f}):\n{doc.text}\n"
            )
        
        return "\n".join(formatted_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        if not self.config.use_context or not context:
            return query
        
        prompt_template = """Based on the following retrieved documents, please answer the question.

Retrieved Documents:
{context}

Question: {query}

Answer:"""
        
        return prompt_template.format(context=context, query=query)
