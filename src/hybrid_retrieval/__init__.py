"""Hybrid retrieval scoring: BM25 + TF-IDF fused with Reciprocal Rank Fusion.

Public API:
    ScoredItem            - dataclass for a scored document
    BM25Scorer            - BM25 keyword scorer over the bm25s library
    INDEX_PATH            - default BM25 index location
    STOPWORDS_FR          - French stopword list
    STOPWORDS_EN          - English stopword list
    STOPWORDS_BILINGUAL   - combined FR+EN stopword list
    reciprocal_rank_fusion - fuse ranked lists with RRF (k=60)
    tfidf_ranked          - TF-IDF cosine-similarity ranking via scikit-learn
"""

from __future__ import annotations

from .bm25 import (
    INDEX_PATH,
    STOPWORDS_BILINGUAL,
    STOPWORDS_EN,
    STOPWORDS_FR,
    BM25Scorer,
)
from .fusion import reciprocal_rank_fusion, tfidf_ranked
from .types import ScoredItem

__version__ = "0.1.0"

__all__ = [
    "INDEX_PATH",
    "STOPWORDS_BILINGUAL",
    "STOPWORDS_EN",
    "STOPWORDS_FR",
    "BM25Scorer",
    "ScoredItem",
    "reciprocal_rank_fusion",
    "tfidf_ranked",
]
