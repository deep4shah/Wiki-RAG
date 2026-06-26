"""
Embeds text chunks via Ollama and stores/retrieves them with ChromaDB.

ChromaDB is used as a lightweight, file-backed vector store so the
whole project runs locally with no external services beyond Ollama.
"""

from __future__ import annotations

import logging

import chromadb
import ollama

from src.chunking import Chunk
from src.config import Config

logger = logging.getLogger(__name__)


def embed_text(text: str, config: Config) -> list[float]:
    """Embed a single piece of text using the configured Ollama model."""
    response = ollama.embeddings(model=config.embedding_model, prompt=text)
    return response["embedding"]


def build_vector_store(chunks: list[Chunk], config: Config) -> chromadb.Collection:
    """
    Embed all chunks and upsert them into a persistent Chroma collection.

    Re-running this on the same chunk_ids is idempotent thanks to
    upsert, so the pipeline can be safely re-executed without
    duplicating vectors.
    """
    client = chromadb.PersistentClient(path=str(config.vector_store_dir))
    collection = client.get_or_create_collection(name=config.collection_name)

    ids, embeddings, metadatas, texts = [], [], [], []
    for chunk in chunks:
        ids.append(chunk.chunk_id)
        embeddings.append(embed_text(chunk.text, config))
        metadatas.append({"source_title": chunk.source_title, "source_url": chunk.source_url})
        texts.append(chunk.text)

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        logger.info("Upserted %d chunks into collection '%s'", len(ids), config.collection_name)

    return collection


def get_collection(config: Config) -> chromadb.Collection:
    """Open the existing persistent collection without re-embedding."""
    client = chromadb.PersistentClient(path=str(config.vector_store_dir))
    return client.get_or_create_collection(name=config.collection_name)
