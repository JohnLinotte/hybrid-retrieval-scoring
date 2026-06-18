# hybrid-retrieval-scoring

Lightweight hybrid document retrieval for Python. It combines two
complementary relevance signals — **BM25** keyword scoring and **TF-IDF**
cosine similarity — and merges their rankings with **Reciprocal Rank Fusion
(RRF)**. Stopword lists ship bilingual (French + English) so mixed-language
corpora work out of the box.

## Why these three pieces

- **BM25** is a strong sparse keyword ranker: it rewards rare query terms and
  saturates term frequency, which makes it robust for short documents and
  exact-match queries. It is provided by the fast [`bm25s`](https://pypi.org/project/bm25s/)
  library and can be built once and persisted to disk.
- **TF-IDF cosine similarity** (via scikit-learn's `TfidfVectorizer`) gives a
  second sparse signal with different normalization characteristics. Where BM25
  and TF-IDF disagree, the fusion step arbitrates.
- **Reciprocal Rank Fusion** merges any number of ranked lists without needing
  to calibrate their raw scores onto a common scale. Each list contributes
  `weight / (k + rank)` per document, with `k = 60` (the standard constant from
  Cormack et al.). Documents that rank high across multiple signals rise to the
  top.

### Why bilingual FR/EN stopwords

Stopwords (function words like *the*, *le*, *of*, *de*) carry little retrieval
signal and inflate term-frequency noise. A corpus that mixes French and English
documents needs both lists removed, otherwise English stopwords pollute French
documents and vice versa. The default `STOPWORDS_BILINGUAL` is the union of a
French and an English list, so a French query against a mixed corpus is not
diluted by untrimmed English function words (and the reverse).

## Install

From PyPI:

```bash
pip install hybrid-retrieval-scoring
```

Or from source (GitHub):

```bash
pip install git+https://github.com/JohnLinotte/hybrid-retrieval-scoring.git
```

Runtime dependencies: `bm25s`, `scikit-learn`, `numpy`.

## Usage

### Build and persist a BM25 index

```python
from hybrid_retrieval import BM25Scorer

corpus = [
    "Le chat dort sur le canapé du salon.",
    "The quick brown fox jumps over the lazy dog.",
    "La météo annonce de la pluie demain matin.",
]

scorer = BM25Scorer()          # bilingual FR+EN stopwords by default
scorer.build_index(corpus)
scorer.save_index()            # written to BM25_INDEX_PATH (default /tmp/hybrid-retrieval/bm25_index)
```

The index location is parameterizable. Pass it to the constructor, or set the
`BM25_INDEX_PATH` environment variable:

```python
from pathlib import Path
scorer = BM25Scorer(index_path=Path("./my_index"))
```

### Score a query

```python
from hybrid_retrieval import BM25Scorer

scorer = BM25Scorer()
scorer.load_index()            # loads the persisted index + corpus
hits = scorer.score("chat canapé", k=5)
# hits == [(0, 1.83), ...]  -> [(doc_index, bm25_score), ...] sorted desc
```

### TF-IDF ranking

```python
from hybrid_retrieval import tfidf_ranked

corpus = [
    "Le chat dort sur le canapé.",
    "The quick brown fox.",
    "La pluie tombe sur la ville.",
]
ranking = tfidf_ranked(corpus, "chat canapé", top_k=3)
# ranking == [0, ...]  -> document indices, best first
```

### Fuse two rankings with RRF

```python
from hybrid_retrieval import BM25Scorer, tfidf_ranked, reciprocal_rank_fusion

corpus = [...]                 # your documents
query = "chat canapé"

bm25 = BM25Scorer()
bm25.build_index(corpus)
bm25_ranking = [idx for idx, _score in bm25.score(query, k=None)]
tfidf_ranking = tfidf_ranked(corpus, query)

fused = reciprocal_rank_fusion(
    [bm25_ranking, tfidf_ranking],
    weights=[1.0, 0.7],        # optional per-signal weights
    k=60,                      # standard RRF constant
)
# fused == [(doc_index, rrf_score), ...] sorted by score descending
```

A document ranked highly by both BM25 and TF-IDF ends up on top, even if no
single signal placed it first.

## Public API

| Symbol | Description |
| --- | --- |
| `ScoredItem` | Dataclass for a scored document (`index`, `score`, `method`, `text`, `source_type`, `priority`). |
| `BM25Scorer` | BM25 scorer with `build_index` / `save_index` / `load_index` / `score`. |
| `reciprocal_rank_fusion` | Merge ranked lists with RRF (`k=60`). |
| `tfidf_ranked` | TF-IDF cosine-similarity ranking. |
| `STOPWORDS_FR`, `STOPWORDS_EN`, `STOPWORDS_BILINGUAL` | Stopword lists. |
| `INDEX_PATH` | Default BM25 index location. |

## License

MIT — see [LICENSE](LICENSE).
