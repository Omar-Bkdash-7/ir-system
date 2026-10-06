import os
import requests
from typing import List
from services.rag.generators.base_generator import BaseGenerator
from services.rag.schemas import ContextDocument, GenerationConfig
from common.logging_config import setup_logging

logger = setup_logging("rag.huggingface_generator")


class HuggingFaceGenerator(BaseGenerator):
    """
    Hugging Face Inference API generator.
    FREE API with decent rate limits.
    Perfect for academic projects.
    
    Get your free token from: https://huggingface.co/settings/tokens
    """
    
    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN environment variable not set. "
                "Get your free token from: https://huggingface.co/settings/tokens"
            )
        
        # Model mapping
        model_mapping = {
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
            "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
            "flan-t5-large": "google/flan-t5-large",
            "flan-ul2": "google/flan-ul2",
            "default": "google/flan-t5-large"
        }
        
        self.model_id = model_mapping.get(config.model.lower(), config.model)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        
        logger.info(f"Initialized HuggingFace generator with model: {self.model_id}")
    
    def generate(self, query: str, context_docs: List[ContextDocument]) -> str:
        """Generate answer using HuggingFace Inference API."""
        
        context = self._format_context(context_docs)
        prompt = self._build_prompt(query, context)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": 0.95,
                "do_sample": True if self.config.temperature > 0 else False
            },
            "options": {
                "wait_for_model": True  # Wait if model is loading
            }
        }
        
        try:
            logger.info(f"Calling HuggingFace API: {self.model_id}")
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 503:
                logger.warning("Model is loading, retrying...")
                # Model is loading, wait and retry
                import time
                time.sleep(20)
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract answer from response
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                answer = result.get('generated_text', '')
            else:
                answer = str(result)
            
            # Clean up the answer
            if prompt in answer:
                answer = answer[len(prompt):].strip()
            
            logger.info("Response generated successfully")
            return answer
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HuggingFace API error: {e.response.status_code} - {e.response.text}")
            
            if e.response.status_code == 401:
                return "Error: Invalid HuggingFace API token. Please check your token."
            elif e.response.status_code == 429:
                return "Error: Rate limit exceeded. Please try again later."
            else:
                return f"Error: API request failed with status {e.response.status_code}"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"
