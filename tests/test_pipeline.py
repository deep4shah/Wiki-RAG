"""
Smoke tests for the Wiki-RAG pipeline.

These deliberately avoid hitting the network or Ollama so they run
fast and don't require the full stack to be set up — they test the
pure-logic pieces (chunking, prompt building) that don't need
external services. Retrieval/generation are exercised manually via
app.py once Ollama is running.
"""

from src.chunking import Chunk, _split_text
from src.config import Config
from src.data_loader import Document
from src.generator import _build_prompt
from src.retriever import RetrievedChunk


def test_split_text_respects_chunk_size():
    text = "A" * 2000
    chunks = _split_text(text, chunk_size=800, overlap=150)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) > 1


def test_split_text_overlap_validation():
    try:
        _split_text("hello world", chunk_size=10, overlap=10)
        assert False, "expected ValueError for overlap >= chunk_size"
    except ValueError:
        pass


def test_chunk_documents_produces_chunks_with_metadata():
    from src.chunking import chunk_documents

    config = Config(chunk_size=50, chunk_overlap=10)
    docs = [Document(title="Test Article", text="word " * 100, url="http://example.com")]
    chunks = chunk_documents(docs, config)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source_title == "Test Article" for c in chunks)


def test_build_prompt_includes_question_and_context():
    config = Config()
    chunks = [
        RetrievedChunk(text="Overfitting happens when a model memorizes noise.",
                        source_title="Overfitting", source_url="http://example.com", distance=0.1)
    ]
    prompt = _build_prompt("What is overfitting?", chunks, config)

    assert "What is overfitting?" in prompt
    assert "Overfitting happens" in prompt
    assert "Overfitting" in prompt  # source title present


def test_build_prompt_respects_max_context_chars():
    config = Config(max_context_chars=50)
    chunks = [
        RetrievedChunk(text="X" * 200, source_title="Doc1", source_url="", distance=0.1),
        RetrievedChunk(text="Y" * 200, source_title="Doc2", source_url="", distance=0.2),
    ]
    prompt = _build_prompt("Q?", chunks, config)

    # Context section should be truncated well below the full 400 chars of chunk text
    assert len(prompt) < 600
