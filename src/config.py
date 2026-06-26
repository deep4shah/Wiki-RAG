"""
Central configuration for the Wiki-RAG pipeline.

Using a single dataclass keeps every tunable parameter (models, chunk
sizes, paths) in one discoverable place instead of scattered magic
numbers across modules.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Data ---
    wikipedia_topics: list[str] = field(default_factory=lambda: [
        "Machine learning",
        "Deep learning",
        "Artificial neural network",
        "Natural language processing",
        "Supervised learning",
        "Unsupervised learning",
        "Reinforcement learning",
        "Convolutional neural network",
        "Recurrent neural network",
        "Transformer (deep learning architecture)",
        "Attention (machine learning)",
        "Large language model",
        "Generative adversarial network",
        "Decision tree learning",
        "Random forest",
        "Gradient boosting",
        "Support vector machine",
        "Logistic regression",
        "Linear regression",
        "Overfitting",
        "Bias-variance tradeoff",
        "Cross-validation (statistics)",
        "Feature engineering",
        "Dimensionality reduction",
        "Principal component analysis",
        "K-means clustering",
        "Hyperparameter optimization",
        "Data preprocessing",
        "Word embedding",
        "BERT (language model)",
        "GPT-3",
        "Diffusion model",
        "Computer vision",
        "Speech recognition",
        "Time series",
        "Anomaly detection",
        "Recommender system",
        "A/B testing",
        "Statistical significance",
        "Bayesian inference",
    ])
    raw_data_dir: Path = Path("data/raw")

    # --- Chunking ---
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150       # overlap between consecutive chunks

    # --- Embeddings / Vector store ---
    embedding_model: str = "nomic-embed-text"
    vector_store_dir: Path = Path("data/chroma_db")
    collection_name: str = "wiki_rag"

    # --- Retrieval ---
    top_k: int = 4

    # --- Generation ---
    llm_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.2
    max_context_chars: int = 6000  # safety cap on prompt size

    # --- Evaluation ---
    eval_questions_path: Path = Path("data/eval_questions.json")
