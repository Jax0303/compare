from __future__ import annotations
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Index:
    def __init__(self, doc_ids: List[str], texts: List[str]):
        self.doc_ids = doc_ids
        tokenized = [((t or "").split()) for t in texts]
        if not any(len(t) for t in tokenized):
            raise ValueError("BM25Index: all documents are empty after tokenization.")
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores((query or "").split())
        idx = np.argsort(scores)[::-1][:k]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]
