"""Builds the chat messages for RAG. Pure functions, no I/O, easy to unit-test.

Documents are numbered by their ORIGINAL position ([1] = context_documents[0]) so the UI can
map citations back to the retrieved list even when some documents are skipped or truncated.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from services.rag.schemas import ContextDocument

NO_ANSWER = "I could not find the answer in the retrieved documents."

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful research assistant inside an Information Retrieval system.\n"
    "Answer the question using ONLY the information from the provided context documents.\n"
    f'- If the documents do not contain the answer, reply exactly: "{NO_ANSWER}"\n'
    "- Provide a direct, natural answer without mentioning document numbers or IDs.\n"
    "- Be concise (at most 5 sentences) and focus on answering the question clearly.\n"
    "- Never invent facts, numbers, or names not found in the context.\n"
    "- Do NOT say 'according to document' or 'in the text' - just give the answer."
)

UNGROUNDED_SYSTEM_PROMPT = "You are a helpful, concise assistant."

MIN_DOC_CHARS = 300


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(" "))
    if boundary < int(limit * 0.6):
        boundary = limit
    return cut[:boundary].rstrip() + " ..."


def build_context_block(documents: Sequence[ContextDocument], max_chars: int) -> str:
    usable = [(i, d) for i, d in enumerate(documents, start=1) if (d.text or "").strip()]
    if not usable:
        return ""
    per_doc = max(MIN_DOC_CHARS, max_chars // len(usable))
    parts: List[str] = []
    used = 0
    for index, doc in usable:
        # Don't include doc_id or index in the context to avoid model mentioning them
        block = f"{_truncate(doc.text, per_doc)}"
        if parts and used + len(block) > max_chars:
            break  # budget exhausted: drop the lowest-ranked remaining documents
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_messages(
    query: str,
    documents: Sequence[ContextDocument],
    *,
    max_context_chars: int,
    system_prompt: Optional[str] = None,
    use_context: bool = True,
) -> List[Dict[str, str]]:
    if not use_context:
        return [
            {"role": "system", "content": (system_prompt or UNGROUNDED_SYSTEM_PROMPT).strip()},
            {"role": "user", "content": query.strip()},
        ]
    context = build_context_block(documents, max_context_chars)
    user = f"Context:\n{context}\n\nQuestion: {query.strip()}\nAnswer:"
    return [
        {"role": "system", "content": (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()},
        {"role": "user", "content": user},
    ]


_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks (reasoning models) and a dangling unclosed <think>."""
    text = _THINK_CLOSED.sub("", text or "")
    text = _THINK_OPEN.sub("", text)
    return text.strip()
