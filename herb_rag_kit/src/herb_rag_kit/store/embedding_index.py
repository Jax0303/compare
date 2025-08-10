from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

class EmbeddingIndex:
    def __init__(self, doc_ids: List[str], vectors: np.ndarray):
        self.doc_ids = doc_ids
        self.vecs = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

    def search(self, qvec: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        q = qvec / (np.linalg.norm(qvec) + 1e-9)
        sims = self.vecs @ q
        idx = np.argsort(sims)[::-1][:k]
        idx = np.ravel(idx)  # 1차원으로 평탄화
        return [(self.doc_ids[int(i)], float(sims[int(i)])) for i in idx]
