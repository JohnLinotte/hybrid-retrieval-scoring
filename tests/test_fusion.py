"""Tests for Reciprocal Rank Fusion and the TF-IDF ranking helper."""

from __future__ import annotations

import math

from hybrid_retrieval import reciprocal_rank_fusion, tfidf_ranked


def test_rrf_item_high_in_both_wins():
    """An item ranked high in both lists ends up on top."""
    list_a = [10, 20, 30]
    list_b = [20, 10, 40]
    fused = reciprocal_rank_fusion([list_a, list_b])

    # Item 20: rank 2 in A, rank 1 in B -> strongest combined position.
    # Item 10: rank 1 in A, rank 2 in B -> same total score as 20.
    fused_ids = [doc for doc, _ in fused]
    assert set(fused_ids[:2]) == {10, 20}
    # 30 and 40 each appear in only one list and rank below the shared pair.
    assert fused_ids[2] in {30, 40}
    assert fused_ids[3] in {30, 40}


def test_rrf_known_math_k60():
    """Verify the exact k=60 RRF arithmetic on a known example."""
    list_a = [1, 2]          # 1 at rank 1, 2 at rank 2
    list_b = [2, 3]          # 2 at rank 1, 3 at rank 2
    fused = dict(reciprocal_rank_fusion([list_a, list_b], k=60))

    # Item 1: only in A at rank 1 -> 1/(60+1)
    assert math.isclose(fused[1], 1 / 61)
    # Item 2: A rank 2 + B rank 1 -> 1/(60+2) + 1/(60+1)
    assert math.isclose(fused[2], 1 / 62 + 1 / 61)
    # Item 3: only in B at rank 2 -> 1/(60+2)
    assert math.isclose(fused[3], 1 / 62)

    # Item 2 has the highest fused score.
    top = max(fused.items(), key=lambda kv: kv[1])[0]
    assert top == 2


def test_rrf_weights_applied():
    """Per-list weights scale each list's contribution."""
    list_a = [1]
    list_b = [2]
    fused = dict(reciprocal_rank_fusion([list_a, list_b], weights=[2.0, 1.0], k=60))
    # Item 1: 2.0/(60+1); Item 2: 1.0/(60+1) -> item 1 wins because of weight.
    assert math.isclose(fused[1], 2.0 / 61)
    assert math.isclose(fused[2], 1.0 / 61)
    assert fused[1] > fused[2]


def test_rrf_empty():
    """Empty input yields an empty result."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_tfidf_ranked_basic():
    """TF-IDF ranks the document that shares query terms first."""
    corpus = [
        "apple pie recipe with butter and flour",
        "the train departs from the station",
        "a cat sleeps on the sofa",
    ]
    ranking = tfidf_ranked(corpus, "apple pie butter")
    assert ranking, "expected non-empty TF-IDF ranking"
    assert ranking[0] == 0


def test_tfidf_ranked_top_k_and_empty():
    """top_k caps the result; empty corpus yields an empty list."""
    corpus = ["alpha beta gamma", "delta epsilon", "alpha delta"]
    ranking = tfidf_ranked(corpus, "alpha", top_k=1)
    assert len(ranking) == 1
    assert tfidf_ranked([], "anything") == []


def test_tfidf_then_rrf_fusion():
    """End-to-end: fuse a TF-IDF ranking with a second ranking via RRF."""
    corpus = [
        "le chat dort sur le canapé",
        "the quick brown fox",
        "la pluie tombe sur la ville",
    ]
    query = "chat canapé"
    tfidf_ranking = tfidf_ranked(corpus, query)
    # A second, independent ranking that also favours document 0.
    other_ranking = [0, 2, 1]
    fused = reciprocal_rank_fusion([tfidf_ranking, other_ranking])
    assert fused[0][0] == 0
