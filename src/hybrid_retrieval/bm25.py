"""BM25 keyword scoring backed by the ``bm25s`` library.

``bm25s`` is a fast, pure-NumPy BM25 implementation (orders of magnitude
faster than ``rank-bm25``). The recommended pattern is to build the index
once and persist it to disk, then load it cheaply wherever scoring happens.

The module ships pre-built bilingual French + English stopword lists so a
mixed-language corpus is handled out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

try:
    import bm25s
    HAS_BM25S = True
except ImportError:
    HAS_BM25S = False

# Default index location. Override per instance via the constructor, or
# globally via the BM25_INDEX_PATH environment variable.
INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", "/tmp/hybrid-retrieval/bm25_index"))

# French stopwords (common words to exclude from BM25 scoring).
STOPWORDS_FR = [
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "en",
    "que", "qui", "dans", "ce", "il", "ne", "sur", "se", "pas", "plus",
    "par", "je", "avec", "tout", "faire", "son", "mais", "nous", "comme",
    "ou", "si", "leur", "y", "dire", "elle", "entre", "quand", "au", "aux",
    "avoir", "etre", "pour", "a", "cette", "mon", "sa", "ses", "ces", "votre",
    "nos", "vos", "sont", "aussi", "bien", "peut", "tous", "apres", "on",
    "meme", "mes", "car", "donc", "ni", "chez", "encore", "dont",
    "lui", "ici", "puis", "vers", "eux", "peu", "tres", "trop", "non",
    "oui", "sans", "sous", "fait", "autre", "cela", "fois", "jour",
]

# English stopwords (standard list, paired with the French one for
# bilingual corpora).
STOPWORDS_EN = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
    "me", "when", "make", "can", "like", "no", "just", "him", "know", "take",
    "into", "your", "some", "could", "them", "than", "other", "been", "its",
    "over", "then", "now", "may", "also", "more", "has", "was", "were", "had",
    "are", "is", "am", "being",
]

# Combined bilingual stopwords (FR + EN), de-duplicated.
STOPWORDS_BILINGUAL = list(set(STOPWORDS_FR + STOPWORDS_EN))


class BM25Scorer:
    """BM25 keyword relevance scorer.

    Usage::

        # Build and persist:
        scorer = BM25Scorer()
        scorer.build_index(texts)
        scorer.save_index()

        # Load and query:
        scorer = BM25Scorer()
        scorer.load_index()
        scores = scorer.score(query, k=10)
    """

    def __init__(self, index_path: Path | None = None, stopwords: str | list[str] | None = None):
        self.index_path = index_path or INDEX_PATH
        # Default to bilingual FR+EN stopwords.
        self.stopwords: str | list[str] = stopwords if stopwords is not None else STOPWORDS_BILINGUAL
        self._retriever = None
        self._corpus_texts: list[str] = []

    def build_index(self, texts: Sequence[str]) -> None:
        """Build a BM25 index from corpus texts.

        Args:
            texts: List of document texts to index.
        """
        if not HAS_BM25S:
            raise ImportError("bm25s not installed. Run: pip install bm25s")

        self._corpus_texts = list(texts)
        if not self._corpus_texts:
            return

        # Tokenize corpus.
        corpus_tokens = bm25s.tokenize(self._corpus_texts, stopwords=self.stopwords)

        # Build index.
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)

    def save_index(self) -> None:
        """Persist the index to disk for fast loading."""
        if self._retriever is None:
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save BM25 retriever together with the corpus.
        self._retriever.save(str(self.index_path), corpus=self._corpus_texts)

    def load_index(self) -> bool:
        """Load a pre-built index from disk.

        Returns:
            True if the index loaded successfully, False otherwise.
        """
        if not HAS_BM25S:
            return False

        if not self.index_path.exists():
            return False

        try:
            self._retriever = bm25s.BM25.load(str(self.index_path), load_corpus=True)
            # bm25s stores the corpus as a list of dicts: [{"id": 0, "text": "..."}, ...]
            raw_corpus = self._retriever.corpus if hasattr(self._retriever, "corpus") else []
            if raw_corpus and isinstance(raw_corpus[0], dict):
                self._corpus_texts = [item.get("text", "") for item in raw_corpus]
            else:
                self._corpus_texts = list(raw_corpus) if raw_corpus else []
            return True
        except (OSError, ValueError, KeyError):
            return False

    def score(self, query: str, k: int | None = None) -> list[tuple[int, float]]:
        """Score documents against a query.

        Args:
            query: Search query.
            k: Number of results to return (None = all).

        Returns:
            List of (doc_index, score) tuples, sorted by score descending.
        """
        if self._retriever is None or not self._corpus_texts:
            return []

        if not HAS_BM25S:
            return []

        k = k or len(self._corpus_texts)

        # Tokenize query.
        query_tokens = bm25s.tokenize([query], stopwords=self.stopwords)

        # Retrieve results.
        results, scores = self._retriever.retrieve(query_tokens, k=min(k, len(self._corpus_texts)))

        # Convert to a list of (index, score) tuples.
        # bm25s returns dicts {"id": int, "text": str} when the corpus was
        # loaded, or plain indices when no corpus is attached.
        scored = []
        for doc_items, doc_scores in zip(results, scores, strict=False):
            for item, sc in zip(doc_items, doc_scores, strict=False):
                if sc > 0:  # Only include positive scores.
                    idx = item.get("id", 0) if isinstance(item, dict) else item
                    scored.append((int(idx), float(sc)))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    @property
    def has_index(self) -> bool:
        """Whether an index is loaded and non-empty."""
        return self._retriever is not None and len(self._corpus_texts) > 0
