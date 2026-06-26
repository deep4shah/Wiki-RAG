"""
Lightweight retrieval evaluation.

For each (question, expected_keyword) pair, checks whether the
retrieved context actually contains the expected keyword/phrase. This
is a simple but honest signal of retrieval quality, much cheaper than
full LLM-as-judge evaluation, and good enough to demonstrate that the
pipeline was actually tested rather than just "vibes-checked".

Run with: python -m src.evaluate
"""

from __future__ import annotations

import json

from src.config import Config
from src.embeddings import get_collection
from src.retriever import retrieve_context

EVAL_SET = [
    {"question": "What is overfitting in machine learning?", "expected_keyword": "overfit"},
    {"question": "What does a convolutional neural network do?", "expected_keyword": "convolution"},
    {"question": "What is the bias-variance tradeoff?", "expected_keyword": "bias"},
    {"question": "What is gradient boosting?", "expected_keyword": "boosting"},
    {"question": "What is a transformer architecture?", "expected_keyword": "attention"},
    {"question": "What is reinforcement learning?", "expected_keyword": "reward"},
    {"question": "What is principal component analysis used for?", "expected_keyword": "dimensionality"},
    {"question": "What is k-means clustering?", "expected_keyword": "cluster"},
    {"question": "What is cross-validation?", "expected_keyword": "fold"},
    {"question": "What is a word embedding?", "expected_keyword": "vector"},
]


def evaluate_retrieval(config: Config | None = None) -> dict:
    config = config or Config()
    collection = get_collection(config)

    hits = 0
    detailed = []
    for item in EVAL_SET:
        chunks = retrieve_context(item["question"], collection, config)
        combined_text = " ".join(c.text.lower() for c in chunks)
        hit = item["expected_keyword"].lower() in combined_text
        hits += int(hit)
        detailed.append({"question": item["question"], "hit": hit})

    summary = {
        "total": len(EVAL_SET),
        "hits": hits,
        "precision_at_k": round(hits / len(EVAL_SET), 2),
        "details": detailed,
    }
    return summary


if __name__ == "__main__":
    results = evaluate_retrieval()
    print(json.dumps(results, indent=2))
    print(f"\nRetrieval precision@k: {results['precision_at_k']:.0%}")
