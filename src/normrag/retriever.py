"""Recuperación léxica BM25 (implementación propia, sin dependencias ni descargas).

Elegir BM25 como línea base es deliberado: en normativa los términos exactos
("garantía de mantenimiento de oferta", "licitación abreviada") pesan mucho, no
requiere modelos descargables y es reproducible. Los embeddings se pueden conectar
implementando la misma interfaz `search`.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .chunking import Chunk
from .text import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    coverage: float  # fracción de términos de la consulta presentes en el fragmento


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("Corpus vacío")
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tokens = [tokenize(c.text) for c in self.chunks]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = np.array([len(t) for t in self._tokens], dtype=float)
        self._avg = float(self._len.mean()) or 1.0
        df: Dict[str, int] = defaultdict(int)
        for tf in self._tf:
            for term in tf:
                df[term] += 1
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._postings: Dict[str, List[int]] = defaultdict(list)
        for i, tf in enumerate(self._tf):
            for term in tf:
                self._postings[term].append(i)

    def _scores(self, q_tokens: List[str]) -> np.ndarray:
        s = np.zeros(len(self.chunks))
        for term in set(q_tokens):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in self._postings[term]:
                tf = self._tf[i][term]
                denom = tf + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s

    def search(self, query: str, k: int = 5) -> List[Hit]:
        q = tokenize(query)
        if not q:
            return []
        scores = self._scores(q)
        order = np.argsort(-scores, kind="stable")[:k]
        qset = set(q)
        hits = []
        for i in order:
            if scores[i] <= 0:
                break
            cov = len(qset & set(self._tf[i])) / len(qset)
            hits.append(Hit(self.chunks[i], float(scores[i]), float(cov)))
        return hits
