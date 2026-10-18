"""BM25 (Okapi) ranking implemented from scratch, with a tokenizer tuned for resumes.

BM25 scores a document by summing, over query terms, an inverse-document-frequency
weight times a saturating, length-normalized term frequency:

    score(D, Q) = sum_t idf(t) * f(t, D) * (k1 + 1) / (f(t, D) + k1 * (1 - b + b * |D| / avgdl))

Lucene-style idf (log(1 + (N - df + 0.5) / (df + 0.5))) keeps weights positive.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was were will with "
    "your you we our can who what which do does using use used based".split()
)


def _stem(token: str) -> str:
    """Light plural stripping so 'pipelines' matches 'pipeline'. Applied to docs and queries alike.

    Only plain words are stemmed; tech tokens such as node.js, c++ and s3 stay intact.
    """
    if not token.isalpha():
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, keep tech tokens like c++, c#, node.js, split ci/cd and full-stack, drop stopwords."""
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower().replace("/", " ").replace("-", " ")):
        token = raw.rstrip(".")
        if len(token) < 2 and token not in {"c", "r"}:
            continue
        if token and token not in STOPWORDS:
            tokens.append(_stem(token))
    return tokens


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_count = len(documents)
        self.doc_len = [len(doc) for doc in documents]
        self.avg_len = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, doc in enumerate(documents):
            for term, freq in Counter(doc).items():
                self.postings.setdefault(term, []).append((index, freq))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> list[float]:
        result = [0.0] * self.doc_count
        for term in set(query_tokens):
            weight = self.idf(term)
            for index, freq in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avg_len)
                result[index] += weight * freq * (self.k1 + 1) / (freq + norm)
        return result

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        """Best `k` (doc_index, score) pairs with score > 0; ties broken by index."""
        ranked = sorted(enumerate(self.scores(query_tokens)), key=lambda item: (-item[1], item[0]))
        return [(index, score) for index, score in ranked[:k] if score > 0]
