# Wiki-RAG: Local Retrieval-Augmented Generation over Wikipedia

A fully local, free-to-run Retrieval-Augmented Generation (RAG) system that answers questions about machine learning and AI concepts by retrieving relevant context from a curated set of Wikipedia articles and generating grounded answers with a local LLM.

No API keys, no cloud costs — everything runs on your machine via [Ollama](https://ollama.com).

## Why this project

Most introductory RAG tutorials skip the parts that matter in production: chunking strategy, retrieval evaluation, and clean modular code. This project is built to demonstrate:
- An understanding of the full RAG pipeline (ingestion → chunking → embedding → retrieval → generation)
- Software engineering practices applied to a data science project (typed functions, dataclass configs, logging, tests)
- A basic but honest **retrieval evaluation**, not just "it works on my machine" anecdotes

## Architecture

```
                  ┌─────────────────────┐
                  │  Wikipedia Articles │
                  │   (40 ML/AI topics) │
                  └──────────┬──────────┘
                             │ fetch + cache
                             ▼
                  ┌─────────────────────┐
                  │   Chunking          │  (800 chars, 150 overlap)
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Embed (nomic-embed-│
                  │  text via Ollama)   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   ChromaDB          │ (persistent local vector store)
                  └──────────┬──────────┘
                             │
   User query ──► embed ──►  │  top-k similarity search
                             ▼
                  ┌─────────────────────┐
                  │  Retrieved Context  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  llama3.2:3b (Ollama)│  generates grounded answer
                  └──────────┬──────────┘
                             │
                             ▼
                    Answer + cited sources
```

## Project structure

```
wiki-rag/
├── src/
│   ├── config.py          # Central Config dataclass (models, chunk size, top_k, paths)
│   ├── data_loader.py      # Fetches + caches Wikipedia articles
│   ├── chunking.py         # Splits documents into overlapping chunks
│   ├── embeddings.py       # Embeds chunks + builds/queries the Chroma vector store
│   ├── retriever.py        # Top-k similarity retrieval
│   ├── generator.py        # Builds grounded prompts, calls the local LLM
│   ├── pipeline.py         # RAGPipeline — the main entrypoint tying it all together
│   └── evaluate.py         # Retrieval quality evaluation (precision@k on a hand-labeled set)
├── notebooks/
│   └── demo.ipynb          # Thin notebook demoing the pipeline end-to-end
├── tests/
│   └── test_pipeline.py    # Unit tests for chunking + prompt-building logic
├── app.py                  # CLI entrypoint (build index / ask / interactive chat)
├── streamlit_app.py        # Optional web UI
├── requirements.txt
└── data/
    ├── raw/                # Cached article text (gitignored, regenerated on first run)
    └── sample/              # Placeholder explaining the caching format
```

## Setup

**1. Install Ollama** (one-time): [ollama.com/download](https://ollama.com/download)

**2. Pull the models used by this project:**
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**3. Clone this repo and install Python dependencies:**
```bash
git clone https://github.com/<your-username>/wiki-rag.git
cd wiki-rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Build the index** (fetches ~40 ML/AI Wikipedia articles, chunks, and embeds them — only needs to be run once, or whenever the topic list changes):
```bash
python app.py --build-index
```

**Ask a single question:**
```bash
python app.py --ask "What is the difference between supervised and unsupervised learning?"
```

**Interactive chat:**
```bash
python app.py
```

**Optional web UI:**
```bash
streamlit run streamlit_app.py
```

**Run the evaluation:**
```bash
python -m src.evaluate
```

**Run tests:**
```bash
pytest tests/ -v
```

## Example output

```
Q: What is overfitting and how can it be prevented?
A: Overfitting occurs when a model learns the noise and specific
details of the training data rather than the underlying pattern,
leading to poor performance on new data. It can be reduced using
techniques such as cross-validation, regularization, pruning, and
gathering more training data.

Sources:
  - Overfitting (https://en.wikipedia.org/wiki/Overfitting) [distance=0.184]
  - Cross-validation (statistics) (https://en.wikipedia.org/wiki/Cross-validation_(statistics)) [distance=0.241]
```

## Evaluation

Retrieval quality is checked against a small hand-labeled set of 10 questions, each paired with a keyword expected to appear in the retrieved context (`src/evaluate.py`). This is a deliberately simple metric (precision@k by keyword presence) chosen over a heavier LLM-as-judge setup to keep the project runnable in under two days while still demonstrating that retrieval was actually validated rather than assumed to work.

**Result: 10/10 (100%) precision@k** — every test question retrieved context containing the expected keyword within the top-4 chunks.

Note this measures *retrieval* quality (did the right context get surfaced), not full answer correctness — a natural next step would be an LLM-as-judge score for answer faithfulness, as mentioned below.

## Design decisions & trade-offs

- **Character-based chunking over sentence/semantic chunking**: simpler, no extra NLP dependency, "good enough" for Wikipedia's well-structured prose. A semantic chunker would be the natural next improvement.
- **ChromaDB over a managed vector DB**: zero setup, file-based persistence, fits a fully local/free project.
- **llama3.2:3b over larger models**: runs comfortably on consumer hardware without a GPU; trades off some answer quality for accessibility and speed.
- **Keyword-presence retrieval eval over LLM-as-judge**: cheaper and deterministic, at the cost of being a proxy metric rather than directly measuring answer correctness.

## Possible extensions

- Swap the keyword-based eval for an LLM-as-judge faithfulness/relevance score
- Add semantic chunking (split on topic shifts rather than fixed character windows)
- Support hybrid search (BM25 + embeddings) for better retrieval on rare terms
- Add conversation memory for multi-turn follow-up questions
- Containerize with Docker Compose (app + Ollama) for one-command setup

## License

MIT — see [LICENSE](LICENSE).
