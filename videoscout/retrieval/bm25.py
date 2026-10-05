"""Okapi BM25, written out so the scoring is easy to read and explain.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

k1 controls term-frequency saturation (the 5th mention of a word adds little);
b controls length normalisation (long documents are not favoured just for being long).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[a-z0-9]+|[一-鿿]")  # English words, or single CJK characters
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "with what which who when where how does did do there their they he she his her you your".split()
)


def tokenize(text: str) -> list[str]:
    return [tok for tok in _WORD.findall(text.lower()) if tok not in _STOP]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) if len(self.docs) and self.lengths.sum() else 1.0
        df: Counter = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([doc.get(term, 0) for doc in self.docs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
