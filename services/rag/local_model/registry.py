"""Vetted local models. All are Apache-2.0, ungated (no license click-through) and instruction-tuned.

NOTE: do NOT use base models such as distilgpt2/gpt2 for RAG: they are not instruction-tuned and
have a 1024-token window, so they cannot follow "answer from these documents".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

ALLOW = ("*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken")
IGNORE = ("*.bin", "*.h5", "*.msgpack", "*.onnx", "*.gguf", "*.pt", "*.ot", "original/*", "onnx/*", "*.md")


@dataclass(frozen=True)
class LocalModelSpec:
    key: str
    repo_id: str
    display_name: str
    approx_size_mb: int
    allow_patterns: Tuple[str, ...] = ALLOW
    ignore_patterns: Tuple[str, ...] = IGNORE


REGISTRY: Dict[str, LocalModelSpec] = {
    s.key: s
    for s in (
        LocalModelSpec("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", 990),
        LocalModelSpec("qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct", 3090),
        LocalModelSpec("smollm2-360m-instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M Instruct", 725),
    )
}


def get_spec(key: str) -> Optional[LocalModelSpec]:
    return REGISTRY.get((key or "").strip().lower())
