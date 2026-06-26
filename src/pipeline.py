"""
High-level RAGPipeline class that ties data loading, chunking,
embedding, retrieval, and generation into a single interface.

This is the main entrypoint other code (notebook, CLI, Streamlit app,
tests) should import and use, rather than calling individual modules
directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.chunking import chunk_documents
from src.config import Config
from src.data_loader import fetch_wikipedia_articles
from src.embeddings import build_vector_store, get_collection
from src.generator import generate_answer
from src.retriever import RetrievedChunk, retrieve_context

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    """Wraps the full ingest -> embed -> retrieve -> generate flow."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._collection = None

    def build_index(self) -> None:
        """One-time (or re-run-as-needed) step: fetch, chunk, and embed the corpus."""
        documents = fetch_wikipedia_articles(self.config)
        chunks = chunk_documents(documents, self.config)
        self._collection = build_vector_store(chunks, self.config)
        logger.info("Index built: %d documents, %d chunks", len(documents), len(chunks))

    def _ensure_collection(self):
        if self._collection is None:
            self._collection = get_collection(self.config)
        return self._collection

    def ask(self, query: str) -> RAGResult:
        """Answer a single question using retrieval-augmented generation."""
        collection = self._ensure_collection()
        context_chunks = retrieve_context(query, collection, self.config)
        answer = generate_answer(query, context_chunks, self.config)
        return RAGResult(answer=answer, sources=context_chunks)
