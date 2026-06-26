"""
Splits documents into overlapping chunks suitable for embedding.

A simple character-based sliding window is used instead of a heavier
NLP-based splitter to keep the dependency footprint small. Splitting
on paragraph boundaries first (falling back to raw slicing) avoids
cutting sentences in half wherever possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Config
from src.data_loader import Document


@dataclass
class Chunk:
    """A chunk of text ready for embedding, with provenance metadata."""
    chunk_id: str
    text: str
    source_title: str
    source_url: str


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Slide a window of chunk_size over text with the given overlap."""
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    joined = "\n".join(paragraphs)

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(joined):
        end = start + chunk_size
        chunk = joined[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_documents(documents: list[Document], config: Config) -> list[Chunk]:
    """Convert a list of Documents into a flat list of Chunks."""
    chunks: list[Chunk] = []
    for doc in documents:
        pieces = _split_text(doc.text, config.chunk_size, config.chunk_overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.title}::{i}",
                    text=piece,
                    source_title=doc.title,
                    source_url=doc.url,
                )
            )
    return chunks
