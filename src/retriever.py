"""
Retrieves the most relevant chunks for a query from the vector store.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from src.config import Config
from src.embeddings import embed_text


@dataclass
class RetrievedChunk:
    text: str
    source_title: str
    source_url: str
    distance: float


def retrieve_context(query: str, collection: chromadb.Collection, config: Config) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks."""
    query_embedding = embed_text(query, config)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config.top_k,
    )

    retrieved: list[RetrievedChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        retrieved.append(
            RetrievedChunk(
                text=doc,
                source_title=meta.get("source_title", "unknown"),
                source_url=meta.get("source_url", ""),
                distance=dist,
            )
        )
    return retrieved
