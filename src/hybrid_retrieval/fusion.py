"""Rank fusion and TF-IDF ranking helpers.

This module provides two building blocks for hybrid retrieval:

* :func:`reciprocal_rank_fusion` merges several ranked lists into one,
  using the standard RRF formula with constant ``k=60``.
* :func:`tfidf_ranked` produces a TF-IDF ranking for a query against a
  corpus, using scikit-learn's :class:`TfidfVectorizer` and cosine
  similarity.

Combine BM25 (see :mod:`hybrid_retrieval.bm25`) and TF-IDF rankings with
RRF to get a robust hybrid signal.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each list contains doc indices ordered by relevance
                      (best first).
        weights: Per-list weight multipliers (default 1.0 each).
        k: RRF constant (default 60, standard value from Cormack et al.).

    Returns:
        [(doc_index, rrf_score)] sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[int, float] = defaultdict(float)
    for weight, ranked in zip(weights, ranked_lists, strict=False):
        for rank, doc_idx in enumerate(ranked, start=1):
            scores[doc_idx] += weight / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# TF-IDF ranking
# ---------------------------------------------------------------------------

def tfidf_ranked(
    corpus: list[str],
    query: str,
    top_k: int | None = None,
) -> list[int]:
    """Rank corpus documents against a query using TF-IDF cosine similarity.

    A :class:`~sklearn.feature_extraction.text.TfidfVectorizer` is fit on the
    corpus, the query is transformed into the same vector space, and documents
    are ranked by their similarity to the query.

    Args:
        corpus: List of document texts.
        query: Search query.
        top_k: Maximum number of doc indices to return (None = all matching).

    Returns:
        Document indices ordered by descending similarity. Documents with a
        zero similarity score (no shared terms) are omitted. Returns an empty
        list when the corpus is empty or the vectorizer yields no vocabulary
        (e.g. every token is filtered out).
    """
    if not corpus:
        return []

    try:
        vectorizer = TfidfVectorizer()
        doc_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query])
    except ValueError:
        # Raised when the corpus contains only stop words / empty vocabulary.
        return []

    # L2-normalized TF-IDF rows mean the dot product equals cosine similarity.
    similarities = (doc_matrix @ query_vec.T).toarray().ravel()

    ranked = np.argsort(similarities)[::-1]
    result = [int(i) for i in ranked if similarities[i] > 0.0]

    if top_k is not None:
        result = result[:top_k]
    return result
