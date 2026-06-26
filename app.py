"""
Simple CLI for the Wiki-RAG pipeline.

Usage:
    python app.py --build-index        # one-time: fetch + embed corpus
    python app.py --ask "What is overfitting?"
    python app.py                      # interactive chat loop
"""

from __future__ import annotations

import argparse

from src.config import Config
from src.pipeline import RAGPipeline


def print_result(result, query: str) -> None:
    print(f"\nQ: {query}")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for src in result.sources:
        print(f"  - {src.source_title} ({src.source_url})  [distance={src.distance:.3f}]")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Wiki-RAG: local Q&A over Wikipedia articles")
    parser.add_argument("--build-index", action="store_true", help="Fetch articles and build the vector index")
    parser.add_argument("--ask", type=str, help="Ask a single question and exit")
    args = parser.parse_args()

    pipeline = RAGPipeline(Config())

    if args.build_index:
        print("Building index (this fetches Wikipedia articles and embeds them)...")
        pipeline.build_index()
        print("Index built. You can now run --ask or use the interactive loop.")
        return

    if args.ask:
        result = pipeline.ask(args.ask)
        print_result(result, args.ask)
        return

    print("Wiki-RAG interactive mode. Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = pipeline.ask(query)
        print_result(result, query)


if __name__ == "__main__":
    main()
