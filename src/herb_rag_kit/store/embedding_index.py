from __future__ import annotations
from typing import List, Tuple
import numpy as np

class EmbeddingIndex:
    def __init__(self, doc_ids: List[str], vectors: np.ndarray):
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(doc_ids):
            raise ValueError(f"EmbeddingIndex: bad shape {vectors.shape} for {len(doc_ids)} docs")
        self.doc_ids = doc_ids
        self.vecs = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

    def search(self, qvec: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        qvec = np.asarray(qvec, dtype="float32").reshape(-1)
        if qvec.shape[0] != self.vecs.shape[1]:
            raise ValueError(f"EmbeddingIndex.search: qvec dim {qvec.shape[0]} != index dim {self.vecs.shape[1]}")
        q = qvec / (np.linalg.norm(qvec) + 1e-9)
        sims = self.vecs @ q
        idx = np.argsort(sims)[::-1][:k]
        return [(self.doc_ids[i], float(sims[i])) for i in idx]
