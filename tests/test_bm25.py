"""BM25 tests over a real bilingual FR+EN corpus (no mocks)."""

from __future__ import annotations

from pathlib import Path

from hybrid_retrieval import BM25Scorer, STOPWORDS_BILINGUAL, STOPWORDS_EN, STOPWORDS_FR

# A small bilingual document set. Indices 0-2 are French, 3-5 are English.
CORPUS = [
    "Le chat dort paisiblement sur le canapé du salon toute la journée.",  # 0 FR
    "La recette de tarte aux pommes demande du beurre et de la farine.",   # 1 FR
    "Le train pour Bruxelles part de la gare à huit heures du matin.",     # 2 FR
    "The cat sleeps peacefully on the living room sofa all day long.",     # 3 EN
    "The apple pie recipe calls for butter and a cup of flour.",           # 4 EN
    "The train to Brussels departs from the station at eight in the morning.",  # 5 EN
]


def _build(tmp_path: Path) -> BM25Scorer:
    scorer = BM25Scorer(index_path=tmp_path / "bm25_index")
    scorer.build_index(CORPUS)
    return scorer


def test_french_query_ranks_french_document_first(tmp_path):
    """A French query must rank the relevant FR document above the EN one."""
    scorer = _build(tmp_path)
    results = scorer.score("chat canapé salon", k=None)

    assert results, "expected non-empty BM25 results"
    top_idx = results[0][0]
    assert top_idx == 0, f"expected FR cat doc (0) on top, got {top_idx}"

    # The French cat document must outrank the English cat document.
    ranking = [idx for idx, _ in results]
    assert ranking.index(0) < ranking.index(3) if 3 in ranking else True


def test_english_query_ranks_english_document_first(tmp_path):
    """Symmetric check: an English query favours the EN document."""
    scorer = _build(tmp_path)
    results = scorer.score("cat sofa living room", k=None)

    assert results
    assert results[0][0] == 3, f"expected EN cat doc (3) on top, got {results[0][0]}"


def test_save_and_load_roundtrip(tmp_path):
    """An index persisted to disk loads back and scores identically."""
    scorer = _build(tmp_path)
    scorer.save_index()
    assert (tmp_path / "bm25_index").exists()

    reloaded = BM25Scorer(index_path=tmp_path / "bm25_index")
    assert reloaded.load_index() is True
    assert reloaded.has_index

    results = reloaded.score("tarte pommes beurre", k=None)
    assert results
    assert results[0][0] == 1, "expected FR apple-pie doc (1) on top after reload"


def test_bilingual_stopwords_cover_both_languages():
    """The bilingual list is the union of the FR and EN stopword lists."""
    assert "le" in STOPWORDS_FR
    assert "the" in STOPWORDS_EN
    assert "le" in STOPWORDS_BILINGUAL
    assert "the" in STOPWORDS_BILINGUAL
    assert set(STOPWORDS_FR) | set(STOPWORDS_EN) == set(STOPWORDS_BILINGUAL)


def test_score_returns_only_positive(tmp_path):
    """Scores are strictly positive and sorted descending."""
    scorer = _build(tmp_path)
    results = scorer.score("train gare Bruxelles", k=None)
    assert results
    scores = [s for _, s in results]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)
