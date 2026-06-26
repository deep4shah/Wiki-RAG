"""
Fetches and caches Wikipedia articles used as the knowledge base.

Articles are cached as plain-text files on disk so repeated runs of the
pipeline don't re-hit the network, and so the corpus is inspectable /
versionable (minus the actual text, which is excluded from git via
.gitignore to keep the repo light).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import wikipedia

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single source document (one Wikipedia article)."""
    title: str
    text: str
    url: str


def _cache_path(raw_dir: Path, title: str) -> Path:
    safe_name = title.replace("/", "_").replace(" ", "_")
    return raw_dir / f"{safe_name}.txt"


def fetch_wikipedia_articles(config: Config) -> list[Document]:
    """
    Fetch each topic in config.wikipedia_topics, using the local cache
    in config.raw_data_dir when available, falling back to the
    Wikipedia API otherwise.

    Topics that fail to resolve (disambiguation, page not found) are
    logged and skipped rather than crashing the whole pipeline.
    """
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []

    for topic in config.wikipedia_topics:
        cache_file = _cache_path(config.raw_data_dir, topic)

        if cache_file.exists():
            text = cache_file.read_text(encoding="utf-8")
            url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            documents.append(Document(title=topic, text=text, url=url))
            continue

        try:
            page = wikipedia.page(topic, auto_suggest=False)
            cache_file.write_text(page.content, encoding="utf-8")
            documents.append(Document(title=page.title, text=page.content, url=page.url))
            logger.info("Fetched and cached: %s", topic)
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning("Skipping '%s': disambiguation page (%s)", topic, e.options[:3])
        except wikipedia.exceptions.PageError:
            logger.warning("Skipping '%s': page not found", topic)
        except Exception as e:  # noqa: BLE001 - log and continue, don't kill the run
            logger.warning("Skipping '%s': unexpected error (%s)", topic, e)

    logger.info("Loaded %d/%d articles", len(documents), len(config.wikipedia_topics))
    return documents
