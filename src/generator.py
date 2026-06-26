"""
Builds a grounded prompt from retrieved context and calls the local
LLM (via Ollama) to generate an answer.
"""

from __future__ import annotations

import ollama

from src.config import Config
from src.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using "
    "the provided context. If the context does not contain enough "
    "information to answer, say so clearly instead of guessing. "
    "Keep answers concise and cite which source(s) you used by title."
)


def _build_prompt(query: str, context_chunks: list[RetrievedChunk], config: Config) -> str:
    context_text = ""
    for chunk in context_chunks:
        addition = f"\n\n[Source: {chunk.source_title}]\n{chunk.text}"
        if len(context_text) + len(addition) > config.max_context_chars:
            break
        context_text += addition

    return (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above:"
    )


def generate_answer(query: str, context_chunks: list[RetrievedChunk], config: Config) -> str:
    """Generate a grounded answer to query using the supplied context chunks."""
    prompt = _build_prompt(query, context_chunks, config)

    response = ollama.chat(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": config.temperature},
    )
    return response["message"]["content"]
