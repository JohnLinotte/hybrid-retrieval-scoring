"""Shared dataclass for scored retrieval results.

Defined in its own module so callers can import ``ScoredItem`` without
pulling in any optional scoring backends.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredItem:
    """A single scored document returned by a retriever.

    Attributes:
        index: Position of the document in the original corpus.
        score: Relevance score (higher is better).
        method: How the score was produced ("bm25", "tfidf", or "rrf").
        text: The document text.
        source_type: Optional free-form label for the document origin.
        priority: Optional caller-supplied weight in the range 0.0-1.0.
    """
    index: int
    score: float
    method: str  # "bm25", "tfidf", or "rrf"
    text: str
    source_type: str = ""
    priority: float = 0.5
