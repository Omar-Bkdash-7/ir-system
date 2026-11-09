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
    "- Support every claim with the bracketed number of the document it comes from, e.g. [1] or [2][3].\n"
    "- Every document in the context starts with its number: [1] is the first retrieved document, [2] the second.\n"
    "- Do NOT say 'according to document' or 'in the text' - just give the answer with its citation.\n"
    "- Be concise (at most 5 sentences) and focus on answering the question clearly.\n"
    "- Never invent facts, numbers, or names not found in the context."
)

UNGROUNDED_SYSTEM_PROMPT = "You are a helpful, concise assistant."

MIN_DOC_CHARS = 300
ELLIPSIS = " ..."
SEPARATOR = "\n\n"


def _truncate(text: str, limit: int) -> str:
    """Cut text to at most `limit` characters, ellipsis included (it used to be appended on top of the limit)."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    room = max(limit - len(ELLIPSIS), 0)
    cut = text[:room]
    boundary = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(" "))
    if boundary < int(room * 0.6):
        boundary = room
    return cut[:boundary].rstrip() + ELLIPSIS


def build_context_block(documents: Sequence[ContextDocument], max_chars: int) -> str:
    usable = [(i, d) for i, d in enumerate(documents, start=1) if (d.text or "").strip()]
    if not usable:
        return ""
    # Distributing the budget over the documents must reserve the separators too, otherwise the joined block is
    # longer than max_chars even though every document stayed inside its own share.
    per_doc = max(MIN_DOC_CHARS, (max_chars - len(SEPARATOR) * (len(usable) - 1)) // len(usable))
    parts: List[str] = []
    used = 0
    for index, doc in usable:
        # The number is the document's position in context_documents, so the UI can map a citation back to the source.
        label = f"[{index}] "
        block = label + _truncate(doc.text, per_doc - len(label))
        cost = len(block) + (len(SEPARATOR) if parts else 0)
        if parts and used + cost > max_chars:
            break  # budget exhausted: drop the lowest-ranked remaining documents
        parts.append(block)
        used += cost
    return SEPARATOR.join(parts)


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


_THINK_CLOSED = re.compile(r" thinking.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r" thinking.*\Z", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove  thinking...</think> blocks (reasoning models) and a dangling unclosed  thinking."""
    text = _THINK_CLOSED.sub("", text or "")
    text = _THINK_OPEN.sub("", text)
    return text.strip()
